/* Exact 2x2 matrix geometry and actual finite NS-5 iteration.
   Original educational implementation inspired by the cited interactive notes.
   All panels share the same coordinate scale; no visual singular-value interpolation. */
(function(){
'use strict';
const mul=(a,b)=>[[a[0][0]*b[0][0]+a[0][1]*b[1][0],a[0][0]*b[0][1]+a[0][1]*b[1][1]],[a[1][0]*b[0][0]+a[1][1]*b[1][0],a[1][0]*b[0][1]+a[1][1]*b[1][1]]];
const transpose=a=>[[a[0][0],a[1][0]],[a[0][1],a[1][1]]];
const rot=t=>[[Math.cos(t),-Math.sin(t)],[Math.sin(t),Math.cos(t)]];
const norm=a=>Math.hypot(...a.flat());
const scale=(a,k)=>a.map(r=>r.map(v=>v*k));
const apply=(m,v)=>[m[0][0]*v[0]+m[0][1]*v[1],m[1][0]*v[0]+m[1][1]*v[1]];
const matrix=(s1,s2,u,v)=>mul(mul(rot(u),[[s1,0],[0,s2]]),transpose(rot(v)));
const polar=(s1,s2,u,v)=>matrix(s1>0?1:0,s2>0?1:0,u,v);
function ns(input,steps){let n=norm(input),x=n?scale(input,1/n):[[0,0],[0,0]];for(let k=0;k<steps;k++){let a=mul(x,transpose(x)),aa=mul(a,a),ax=mul(a,x),aax=mul(aa,x);x=x.map((r,i)=>r.map((v,j)=>3.4445*v-4.7750*ax[i][j]+2.0315*aax[i][j]));}return x;}
function singular(m){const a=mul(transpose(m),m),tr=a[0][0]+a[1][1],disc=Math.hypot(a[0][0]-a[1][1],2*a[0][1]);return [Math.sqrt(Math.max(0,(tr+disc)/2)),Math.sqrt(Math.max(0,(tr-disc)/2))];}
const api={mul,transpose,rot,norm,scale,apply,matrix,polar,ns,singular};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
if(typeof document==='undefined')return;
const root=document.querySelector('[data-muon-geometry]');if(!root)return;
const q=s=>root.querySelector(s),s1=q('[data-s1]'),s2=q('[data-s2]'),angle=q('[data-u-angle]'),play=q('[data-geo-play]'),budget=q('[data-ns-step]');
let phase=0,running=false,frame=0,last=0;
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent,txt){let x=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs||{}))x.setAttribute(k,String(v));if(txt!==undefined)x.textContent=txt;if(parent)parent.appendChild(x);return x;}
function segment(svg,a,b,c,w=1,dash){let attrs={x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:c,'stroke-width':w};if(dash)attrs['stroke-dasharray']=dash;el('line',attrs,svg);}
function vector(svg,p,col,label){const o=[240,170],end=[240+62*p[0],170-62*p[1]],len=Math.hypot(...p);if(len<1e-9)return;segment(svg,o,end,col,2.3);const a=Math.atan2(end[1]-170,end[0]-240);el('path',{d:`M${end[0]},${end[1]} L${end[0]-8*Math.cos(a)+3*Math.sin(a)},${end[1]-8*Math.sin(a)-3*Math.cos(a)} L${end[0]-8*Math.cos(a)-3*Math.sin(a)},${end[1]-8*Math.sin(a)+3*Math.cos(a)} Z`,fill:col},svg);el('text',{x:end[0]+(p[0]>=0?10:-10),y:end[1]+(p[1]>=0?-10:20),'text-anchor':p[0]>=0?'start':'end',fill:col,'font-size':15},svg,label);}
function draw(svg,m,kind){svg.replaceChildren();const px=v=>[240+62*v[0],170-62*v[1]];segment(svg,[58,170],[422,170],'#E1E8EE');segment(svg,[240,24],[240,316],'#E1E8EE');for(let i of [-2,-1,1,2]){el('text',{x:240+62*i,y:187,'text-anchor':'middle',fill:'#92A0AE','font-size':11},svg,i);}
el('circle',{cx:240,cy:170,r:62,fill:'none',stroke:'#B9C5D0','stroke-width':1.2,'stroke-dasharray':'4 5'},svg);
let points=[];for(let i=0;i<=180;i++){let t=i*Math.PI*2/180;points.push(px(apply(m,[Math.cos(t),Math.sin(t)])));}el('path',{d:points.map((p,i)=>(i?'L':'M')+p[0].toFixed(2)+','+p[1].toFixed(2)).join(' ')+' Z',fill:'#EAF1F7',stroke:'#789BB9','stroke-width':1.7,'fill-opacity':.55},svg);
for(let i=0;i<12;i++){let t=i*Math.PI/6+phase,p=px(apply(m,[Math.cos(t),Math.sin(t)]));el('circle',{cx:p[0],cy:p[1],r:i===0?5.8:4.1,fill:`hsl(${215+i*13},57%,${i===0?43:54}%)`,stroke:'white','stroke-width':1.5},svg);}
let v=20*Math.PI/180,basis=[[Math.cos(v),Math.sin(v)],[-Math.sin(v),Math.cos(v)]];
vector(svg,apply(m,basis[0]),'#0F4D92',kind==='raw'?'σ₁u₁':kind==='polar'?'u₁':'Xⱼv₁');vector(svg,apply(m,basis[1]),'#42949E',kind==='raw'?'σ₂u₂':kind==='polar'?'u₂':'Xⱼv₂');
el('text',{x:240,y:331,'text-anchor':'middle',fill:'#6B7C8D','font-size':12},svg,'Dashed circle: unit reference · same scale');}
function render(){let a=+s1.value,b=+s2.value;if(b>a){b=a;s2.value=b;}s2.max=String(a);const u=+angle.value*Math.PI/180,v=20*Math.PI/180,g=matrix(a,b,u,v),p=polar(a,b,u,v),step=+budget.value,x=ns(g,step),sv=singular(x);
q('[data-s1-value]').textContent=a.toFixed(2);q('[data-s2-value]').textContent=b.toFixed(2);q('[data-u-value]').textContent=angle.value+'°';q('[data-raw-values]').textContent=`(${a.toFixed(2)}, ${b.toFixed(2)})`;q('[data-polar-values]').textContent=b>0?'(1, 1)':'(1, 0)';q('[data-ns-values]').textContent=`(${sv[0].toFixed(3)}, ${sv[1].toFixed(3)})`;q('[data-ns-index]').textContent=String(step);q('[data-ns-next]').textContent=step===5?'Restart at X₀':'Next inner step';
draw(q('[data-raw-canvas]'),g,'raw');draw(q('[data-polar-canvas]'),p,'polar');draw(q('[data-ns-canvas]'),x,'ns');}
function stop(){running=false;cancelAnimationFrame(frame);last=0;play.textContent='Trace the circle';play.setAttribute('aria-pressed','false');}
function tick(now){if(!running)return;if(last)phase+=(now-last)/5500*Math.PI*2;last=now;render();frame=requestAnimationFrame(tick);}
play.addEventListener('click',()=>{if(running)stop();else{running=true;play.textContent='Pause';play.setAttribute('aria-pressed','true');frame=requestAnimationFrame(tick);}});
for(const x of[s1,s2,angle,budget])x.addEventListener('input',render);
q('[data-ns-next]').addEventListener('click',()=>{budget.value=(+budget.value+1)%6;render();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
if(typeof IntersectionObserver!=='undefined')new IntersectionObserver(es=>{if(!es[0].isIntersecting)stop();}).observe(root);
q('[data-geo-controls]').hidden=false;q('[data-ns-controls]').hidden=false;render();
})();
