// Same definitions as rankability_metrics.py. score: higher = better; dist: final distance (m).
function ranks(a){const i=a.map((v,k)=>k).sort((x,y)=>a[x]-a[y]);const r=new Array(a.length);for(let s=0;s<i.length;){let e=s;while(e+1<i.length&&a[i[e+1]]===a[i[s]])e++;const v=(s+e)/2+1;for(let k=s;k<=e;k++)r[i[k]]=v;s=e+1;}return r;}
function pearson(x,y){const n=x.length;let mx=0,my=0;for(let i=0;i<n;i++){mx+=x[i];my+=y[i];}mx/=n;my/=n;let a=0,b=0,c=0;for(let i=0;i<n;i++){const dx=x[i]-mx,dy=y[i]-my;a+=dx*dy;b+=dx*dx;c+=dy*dy;}return b&&c?a/Math.sqrt(b*c):NaN;}
function spearman(s,d){return pearson(ranks(s),ranks(d.map(v=>-v)));}
function auroc(s,pos){const n1=pos.filter(Boolean).length,n0=pos.length-n1;if(!n1||!n0)return NaN;const r=ranks(s);let t=0;pos.forEach((p,i)=>{if(p)t+=r[i];});return (t-n1*(n1+1)/2)/(n1*n0);}
function pairwise(s,d){let ok=0,tie=0,m=0;for(let i=0;i<s.length;i++)for(let j=0;j<s.length;j++){const dd=d[j]-d[i];if(dd===0)continue;m++;const ss=s[i]-s[j];if(ss===0)tie++;else if(Math.sign(ss)===Math.sign(dd))ok++;}return (ok+0.5*tie)/m;}
function anchorMetrics(s,d){const order=d.map((v,k)=>k).sort((a,b)=>d[a]-d[b]);let top=0;s.forEach((v,k)=>{if(v>s[top])top=k;});const mn=d[order[0]];
 return {spearman:spearman(s,d),auroc_5cm:auroc(s,d.map(v=>v<0.05)),regret_cm:100*(d[top]-mn),hit1:+(top===order[0]),hit3:+order.slice(0,3).includes(top),pairwise:pairwise(s,d)};}
function sig(x){return 1/(1+Math.exp(-x));}
function eceHeldout(S,D,tau=0.05,bins=10){const z=S.map(s=>{const m=s.reduce((a,b)=>a+b,0)/s.length;const sd=Math.sqrt(s.reduce((a,b)=>a+(b-m)**2,0)/s.length)+1e-9;return s.map(v=>(v-m)/sd);});
 const fx=[],fy=[],ex=[],ey=[];z.forEach((r,i)=>r.forEach((v,j)=>{(i%2?ex:fx).push(v);(i%2?ey:fy).push(D[i][j]<tau?1:0);}));
 let w=[0,0];for(let it=0;it<50;it++){let g0=0,g1=0,h00=0,h01=0,h11=0;for(let k=0;k<fx.length;k++){const p=sig(w[0]*fx[k]+w[1]),e=p-fy[k],q=p*(1-p)+1e-9;g0+=e*fx[k];g1+=e;h00+=q*fx[k]*fx[k];h01+=q*fx[k];h11+=q;}const det=h00*h11-h01*h01;if(!det)break;w=[w[0]-(h11*g0-h01*g1)/det,w[1]-(h00*g1-h01*g0)/det];}
 const cnt=new Array(bins).fill(0),sp=new Array(bins).fill(0),sy=new Array(bins).fill(0);ex.forEach((v,k)=>{const p=sig(w[0]*v+w[1]);const b=Math.min(bins-1,Math.floor(p*bins));cnt[b]++;sp[b]+=p;sy[b]+=ey[k];});
 let e=0;for(let b=0;b<bins;b++)if(cnt[b])e+=cnt[b]/ex.length*Math.abs(sp[b]/cnt[b]-sy[b]/cnt[b]);return e;}
function evaluate(S,D,SRC){const mean=(L,k)=>{const v=L.map(q=>q[k]).filter(x=>!isNaN(x));return v.reduce((a,b)=>a+b,0)/v.length;};
 const all=[],fine=[],rnd=[];S.forEach((s,i)=>{const d=D[i],g=SRC[i].map(v=>v>0);all.push(anchorMetrics(s,d));fine.push(anchorMetrics(s.filter((_,k)=>g[k]),d.filter((_,k)=>g[k])));rnd.push(anchorMetrics(s.filter((_,k)=>!g[k]),d.filter((_,k)=>!g[k])));});
 return {n_anchors:S.length,overall_spearman:mean(all,"spearman"),fine_spearman:mean(fine,"spearman"),random_spearman:mean(rnd,"spearman"),auroc_5cm:mean(all,"auroc_5cm"),regret_cm:mean(all,"regret_cm"),hit1:mean(all,"hit1"),hit3:mean(all,"hit3"),fine_pairwise:mean(fine,"pairwise"),ece:eceHeldout(S,D)};}
