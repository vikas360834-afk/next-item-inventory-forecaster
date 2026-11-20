"""Synthetic retail basket data, SKU catalogue, n-gram next-item model and evaluation."""
from collections import Counter, defaultdict
import random

PRODUCTS={"SKU001":"Milk 1L","SKU002":"Bread","SKU003":"Eggs 12-pack","SKU004":"Butter","SKU005":"Cheese","SKU006":"Bananas","SKU007":"Apples","SKU008":"Cereal","SKU009":"Coffee","SKU010":"Tea","SKU011":"Pasta","SKU012":"Tomato Sauce","SKU013":"Rice 5kg","SKU014":"Cooking Oil","SKU015":"Sugar","SKU016":"Flour","SKU017":"Chicken","SKU018":"Yogurt","SKU019":"Orange Juice","SKU020":"Dish Soap"}
DEPARTMENTS={"Dairy":["SKU001","SKU003","SKU004","SKU005","SKU018"],"Bakery":["SKU002","SKU016"],"Fruit":["SKU006","SKU007"],"Breakfast":["SKU008","SKU009","SKU010","SKU019"],"Pantry":["SKU011","SKU012","SKU013","SKU014","SKU015"],"Meat":["SKU017"],"Household":["SKU020"]}
PATTERNS=[["SKU001","SKU002","SKU003","SKU004"],["SKU008","SKU001","SKU006","SKU019"],["SKU011","SKU012","SKU005"],["SKU013","SKU014","SKU017"],["SKU009","SKU002","SKU004"],["SKU006","SKU007","SKU018"],["SKU016","SKU015","SKU003"],["SKU013","SKU017","SKU020"],["SKU010","SKU002","SKU018"],["SKU011","SKU012","SKU017","SKU005"]]

def generate_mock_transactions(n=1200,seed=42):
    if n<1: raise ValueError("n must be positive")
    rng=random.Random(seed); pool=list(PRODUCTS); rows=[]
    for _ in range(n):
        if rng.random()<.78:
            row=list(rng.choice(PATTERNS))
            for _ in range(rng.randint(0,2)):
                sku=rng.choice(pool)
                if sku not in row: row.insert(rng.randrange(len(row)+1),sku)
            rng.shuffle(row)
        else: row=rng.sample(pool,rng.randint(2,6))
        rows.append(row)
    return rows

class SKUSequenceLanguageModel:
    def __init__(self,max_context=3,alpha=.15):
        self.max_context=max_context; self.alpha=alpha; self.counts={i:defaultdict(Counter) for i in range(max_context+1)}; self.unigram=Counter(); self.vocabulary=[]; self.total_tokens=0; self.is_trained=False
    def fit(self,transactions):
        rows=[list(x) for x in transactions if x]
        if not rows: raise ValueError("Training transactions cannot be empty")
        vocab=set()
        for row in rows:
            for i,target in enumerate(row):
                vocab.add(target); self.unigram[target]+=1; self.total_tokens+=1
                for n in range(self.max_context+1): self.counts[n][tuple(row[max(0,i-n):i]) if n else ()][target]+=1
        self.vocabulary=sorted(vocab); self.is_trained=True; return self
    def predict_next(self,basket,top_k=5,exclude_seen=True):
        if not self.is_trained: raise RuntimeError("Train the model first")
        unknown=set(basket)-set(self.vocabulary)
        if unknown: raise ValueError("Unknown SKU(s): "+", ".join(sorted(unknown)))
        dist=None
        for n in range(min(self.max_context,len(basket)),-1,-1):
            c=self.counts[n].get(tuple(basket[-n:]) if n else ())
            if c:
                total=sum(c.values()); denom=total+self.alpha*len(self.vocabulary)
                dist={s:(c[s]+self.alpha)/denom for s in self.vocabulary}; break
        if dist is None: dist={s:1/len(self.vocabulary) for s in self.vocabulary}
        ranked=sorted(((s,p) for s,p in dist.items() if not exclude_seen or s not in basket),key=lambda z:(-z[1],z[0]))[:top_k]
        denom=sum(p for _,p in ranked) or 1
        return [{"sku":s,"name":PRODUCTS.get(s,s),"probability":p/denom,"raw_probability":p} for s,p in ranked]
    def to_dict(self):
        return {"max_context":self.max_context,"alpha":self.alpha,"vocabulary":self.vocabulary,"counts":{str(n):{"|".join(k):dict(v) for k,v in d.items()} for n,d in self.counts.items()}}
    @classmethod
    def from_dict(cls,obj):
        m=cls(obj["max_context"],obj["alpha"]); m.vocabulary=obj["vocabulary"]
        for n,contexts in obj["counts"].items():
            for k,v in contexts.items(): m.counts[int(n)][tuple(k.split("|")) if k else ()]=Counter(v)
        m.unigram=Counter(m.counts[0].get((),{})); m.total_tokens=sum(m.unigram.values()); m.is_trained=True; return m

def milestone_1(n=1200,seed=42):
    rows=generate_mock_transactions(n,seed); return {"products":PRODUCTS.copy(),"transactions":rows,"transaction_count":len(rows),"vocabulary":sorted({s for r in rows for s in r})}
def milestone_2(transactions=None):
    return SKUSequenceLanguageModel().fit(transactions or generate_mock_transactions())
def milestone_3(model,basket,top_k=5): return model.predict_next(basket,top_k)
def popularity_baseline(rows,basket,top_k=5):
    c=Counter(s for r in rows for s in r if s not in basket); return [s for s,_ in sorted(c.items(),key=lambda z:(-z[1],z[0]))[:top_k]]
def evaluate_model(model,test_transactions,top_k=5):
    hits=base_hits=0; rr=[]; total=0; popularity=sorted(model.unigram,key=lambda s:(-model.unigram[s],s))
    for row in test_transactions:
        if len(row)<2 or row[-1] in row[:-1]: continue
        total+=1; target=row[-1]; pred=[x["sku"] for x in model.predict_next(row[:-1],top_k)]
        if target in pred: hits+=1; rr.append(1/(pred.index(target)+1))
        else: rr.append(0)
        base=[s for s in popularity if s not in row[:-1]][:top_k]; base_hits+=target in base
    return {"examples":total,"top_k":top_k,"model_hit_rate":hits/total if total else 0,"model_mrr":sum(rr)/total if total else 0,"popularity_hit_rate":base_hits/total if total else 0}
def milestone_5(seed=42,n=1200,top_k=5):
    rows=generate_mock_transactions(n,seed); random.Random(seed+1).shuffle(rows); split=int(.8*len(rows)); model=milestone_2(rows[:split]); return evaluate_model(model,rows[split:],top_k)
