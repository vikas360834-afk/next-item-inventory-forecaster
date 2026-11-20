"""Compact causal Transformer for next-SKU prediction."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

class SKUTransformer(nn.Module):
    def __init__(self,vocab_size,d_model=64,nhead=4,layers=2,max_len=32):
        super().__init__()
        self.token=nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.position=nn.Embedding(max_len,d_model)
        block=nn.TransformerEncoderLayer(d_model,nhead,d_model*4,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab_size)
    def forward(self,ids):
        pos=torch.arange(ids.size(1),device=ids.device).unsqueeze(0)
        x=self.token(ids)+self.position(pos)
        mask=torch.triu(torch.ones(ids.size(1),ids.size(1),device=ids.device,dtype=torch.bool),diagonal=1)
        x=self.encoder(x,mask=mask,src_key_padding_mask=ids.eq(0))
        return self.head(self.norm(x))

class TransformerForecaster:
    def __init__(self,products,max_len=32):
        self.products=dict(products); self.skus=sorted(products); self.max_len=max_len
        self.stoi={"<PAD>":0,"<BOS>":1,"<EOS>":2}
        self.stoi.update({sku:i+3 for i,sku in enumerate(self.skus)})
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model=SKUTransformer(len(self.stoi),max_len=max_len).to(self.device)
        self.trained=False
    def fit(self,transactions,epochs=20,batch_size=64,lr=.002,seed=42):
        if not transactions: raise ValueError("No transactions supplied")
        torch.manual_seed(seed); rows=[]
        for row in transactions:
            ids=[self.stoi[s] for s in row if s in self.stoi][:self.max_len-1]
            if not ids: continue
            x=[1]+ids; y=ids+[2]
            x=x[:self.max_len]+[0]*max(0,self.max_len-len(x))
            y=y[:self.max_len]+[-100]*max(0,self.max_len-len(y))
            rows.append((x,y))
        if not rows: raise ValueError("No known SKU tokens")
        xs=torch.tensor([r[0] for r in rows]); ys=torch.tensor([r[1] for r in rows])
        loader=DataLoader(TensorDataset(xs,ys),batch_size=batch_size,shuffle=True)
        opt=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=.01)
        loss_fn=nn.CrossEntropyLoss(ignore_index=-100); history=[]
        for _ in range(max(1,epochs)):
            self.model.train(); vals=[]
            for bx,by in loader:
                bx,by=bx.to(self.device),by.to(self.device); opt.zero_grad(set_to_none=True)
                logits=self.model(bx); loss=loss_fn(logits.reshape(-1,logits.size(-1)),by.reshape(-1))
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),1.0); opt.step(); vals.append(float(loss.detach().cpu()))
            history.append(sum(vals)/len(vals))
        self.trained=True; return history
    @torch.no_grad()
    def predict_next(self,basket,top_k=5,exclude_seen=True):
        if not self.trained: raise RuntimeError("Train the Transformer first")
        unknown=set(basket)-set(self.skus)
        if unknown: raise ValueError("Unknown SKU(s): "+", ".join(sorted(unknown)))
        ids=[1]+[self.stoi[s] for s in basket][-(self.max_len-1):]
        self.model.eval(); x=torch.tensor([ids],device=self.device)
        probs=torch.softmax(self.model(x)[0,-1],dim=-1).cpu()
        candidates=[(s,float(probs[self.stoi[s]])) for s in self.skus if not exclude_seen or s not in basket]
        chosen=sorted(candidates,key=lambda z:(-z[1],z[0]))[:top_k]; denom=sum(v for _,v in chosen) or 1
        return [{"sku":s,"name":self.products[s],"probability":v/denom,"raw_probability":v} for s,v in chosen]
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state":self.model.state_dict(),"products":self.products,"max_len":self.max_len},path)
    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location="cpu",weights_only=False); obj=cls(p["products"],p["max_len"])
        obj.model.load_state_dict(p["state"]); obj.trained=True; return obj
