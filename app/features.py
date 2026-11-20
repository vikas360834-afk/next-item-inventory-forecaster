"""Inventory planning, association metrics and CSV transaction import."""
from collections import Counter
import csv,io,random
from app.core import PRODUCTS,DEPARTMENTS

DEPT={s:d for d,items in DEPARTMENTS.items() for s in items}
def build_inventory(transactions,seed=17):
    rng=random.Random(seed); counts=Counter(s for r in transactions for s in r); out=[]
    for sku,name in PRODUCTS.items():
        daily=counts[sku]/max(1,len(transactions)); stock=rng.randint(4,95); point=max(5,round(daily*5)); target=max(15,round(daily*18))
        out.append({"sku":sku,"product":name,"department":DEPT.get(sku,"Other"),"stock_on_hand":stock,"units_sold_30d":counts[sku],"avg_daily_demand":round(daily,2),"reorder_point":point,"target_stock":target,"suggested_reorder":max(0,target-stock),"status":"Reorder" if stock<=point else "Watch" if stock<=point*1.5 else "Healthy"})
    return out
def inventory_report(transactions,inventory):
    counts=Counter(s for r in transactions for s in r); result=[]
    for x in inventory:
        daily=counts[x["sku"]]/max(1,len(transactions)); result.append({**x,"units_sold":counts[x["sku"]],"days_of_cover":round(x["stock_on_hand"]/daily,1) if daily else 999.0,"suggested_reorder":max(0,x["target_stock"]-x["stock_on_hand"])})
    return result
def association_report(transactions,min_pair_count=2):
    singles=Counter(); pairs=Counter()
    for row in transactions:
        u=sorted(set(row)); singles.update(u)
        for i,a in enumerate(u):
            for b in u[i+1:]: pairs[(a,b)]+=1
    n=max(1,len(transactions)); out=[]
    for (a,b),count in pairs.items():
        if count<min_pair_count: continue
        support=count/n; out.append({"Product A":PRODUCTS[a],"Product B":PRODUCTS[b],"Pair Count":count,"Support":round(support,3),"Confidence A→B":round(count/singles[a],3),"Confidence B→A":round(count/singles[b],3),"Lift":round(support/((singles[a]/n)*(singles[b]/n)),2)})
    return sorted(out,key=lambda x:(-x["Lift"],-x["Pair Count"]))
def department_summary(transactions):
    c=Counter(DEPT.get(s,"Other") for r in transactions for s in r); return [{"Department":d,"Units":v} for d,v in c.most_common()]
def transactions_from_csv(raw):
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not reader.fieldnames or not {"transaction_id","sku"}.issubset(reader.fieldnames): raise ValueError("CSV must contain transaction_id and sku columns")
    groups={}
    for r in reader:
        tid=(r.get("transaction_id") or "").strip(); sku=(r.get("sku") or "").strip()
        if not tid or not sku: continue
        if sku not in PRODUCTS: raise ValueError(f"Unknown SKU: {sku}")
        groups.setdefault(tid,[]).append(sku)
    rows=[v for v in groups.values() if len(v)>=2]
    if not rows: raise ValueError("No valid transactions with at least two products")
    return rows
