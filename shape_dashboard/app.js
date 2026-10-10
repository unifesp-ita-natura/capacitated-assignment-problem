"use strict";
const $ = id => document.getElementById(id);
const colors = ["#087d87", "#db783c", "#7a5cbc", "#4a8e50", "#b14f82"];
const fmt = (value, digits=2) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString("pt-BR", {maximumFractionDigits:digits});
let runOptions = null, currentView = null, ticket = 0;

async function request(path, query={}) {
  const response = await fetch(path + "?" + new URLSearchParams(query));
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Não foi possível ler os resultados.");
  return result;
}
function message(text, error=false) { $("message").textContent=text; $("message").className=error?"error":""; }
function choices(id, values, all=null) {
  const select=$(id), previous=select.value; select.replaceChildren();
  const entries=all===null?values:[["",all],...values.map(v=>[v,v])];
  for(const entry of entries) { const [value,label]=Array.isArray(entry)?entry:[entry,entry]; select.add(new Option(label,value)); }
  if([...select.options].some(o=>o.value===previous)) select.value=previous;
}
function query() { return {run:$("run").value, cycle:$("cycle").value, origin:$("origin").value, sector:$("sector").value.trim(), model:$("model").value}; }
const metricColumns = [["candidate","Modelo"],["share_mae_pp","Share MAE (p.p.)"],["total_variation","Variação total"],["mae","MAE (itens)"],["rmse","RMSE (itens)"],["wmape","WMAPE (%)"],["bias","Bias (itens)"],["n_sector_cycles","Curvas"],["n_days","Dias"],["n_uniform_fallback_cycles","Fallbacks"]];
function table(id, rows, columns) {
  const head=document.createElement("thead"), tr=document.createElement("tr");
  for(const [,label] of columns) { const th=document.createElement("th"); th.textContent=label; tr.append(th); }
  head.append(tr); const body=document.createElement("tbody");
  for(const row of rows) { const line=document.createElement("tr"); for(const [key] of columns) { const cell=document.createElement("td"), value=row[key]; cell.textContent= typeof value==="number" ? fmt(key==="wmape"?100*value:value,["share_pred","actual_share"].includes(key)?4:2) : value??"—"; line.append(cell); } body.append(line); }
  $(id).replaceChildren(head,body);
}
function svgElement(name, attrs={}, text=null) {
  const node=document.createElementNS("http://www.w3.org/2000/svg",name);
  for(const [key,value] of Object.entries(attrs)) node.setAttribute(key,value);
  if(text!==null) node.textContent=text; return node;
}
function chart(id, traces, title, suffix="") {
  const container=$(id); container.replaceChildren();
  const valid=traces.flatMap(t=>t.points.map(p=>p.value)).filter(v=>v!=null&&Number.isFinite(v));
  if(!valid.length) { container.textContent="Sem valores definidos neste recorte."; return; }
  const days=[...new Set(traces.flatMap(t=>t.points.map(p=>p.day)))].sort((a,b)=>a-b);
  const width=800,height=280,left=70,right=20,top=20,bottom=50;
  const minimum=Math.min(0,...valid), maximum=Math.max(0,...valid), span=maximum-minimum||1;
  const x=day=>left+(day-days[0])*(width-left-right)/Math.max(1,days[days.length-1]-days[0]);
  const y=value=>height-bottom-(value-minimum)*(height-top-bottom)/span;
  const svg=svgElement("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":title});
  svg.append(svgElement("title",{},title));
  for(let i=0;i<=4;i++) { const value=minimum+span*i/4; svg.append(svgElement("line",{x1:left,y1:y(value),x2:width-right,y2:y(value),stroke:"#e4eaee"})); svg.append(svgElement("text",{x:left-10,y:y(value)+4,"text-anchor":"end",fill:"#687a89","font-size":11},fmt(value,1)+suffix)); }
  if(minimum<0) svg.append(svgElement("line",{x1:left,x2:width-right,y1:y(0),y2:y(0),stroke:"#8da0ad","stroke-dasharray":"4 4"}));
  const stride=Math.max(1,Math.ceil(days.length/6));
  days.forEach((day,i)=>{ if(i%stride===0||i===days.length-1) svg.append(svgElement("text",{x:x(day),y:height-20,"text-anchor":"middle",fill:"#687a89","font-size":11},`Dia ${day}`)); });
  const legend=document.createElement("div"); legend.className="legend";
  for(const trace of traces) {
    const points=trace.points.filter(p=>p.value!=null&&Number.isFinite(p.value));
    svg.append(svgElement("polyline",{points:points.map(p=>`${x(p.day)},${y(p.value)}`).join(" "),fill:"none",stroke:trace.color,"stroke-width":trace.name==="Real"?3:2}));
    for(const point of points) { const circle=svgElement("circle",{cx:x(point.day),cy:y(point.value),r:3,fill:trace.color}); circle.append(svgElement("title",{},`${trace.name} · Dia ${point.day}: ${fmt(point.value)}${suffix}`)); svg.append(circle); }
    const item=document.createElement("span"), swatch=document.createElement("i"); swatch.className="swatch"; swatch.style.background=trace.color; item.append(swatch,document.createTextNode(trace.name)); legend.append(item);
  }
  container.append(svg,legend);
}
function plots(data) {
  const models=Object.keys(data.series), base=data.series[models[0]], percentages=$("unit").value==="pct";
  const dailyKey=percentages?"pred_pct":"items_pred", actualKey=percentages?"actual_pct":"actual";
  const points=(rows,key)=>rows.map(row=>({day:row.relative_day,value:row[key]}));
  const predicted=key=>models.map((name,i)=>({name,color:colors[i%colors.length],points:points(data.series[name],key)}));
  chart("daily",[{name:"Real",color:"#233746",points:points(base,actualKey)},...predicted(dailyKey)],"Real e previsto por dia",percentages?"%":"");
  chart("cumulative",[{name:"Real",color:"#233746",points:points(base,"actual_cumulative_pct")},...predicted("pred_cumulative_pct")],"Participação acumulada","%");
  chart("errors",predicted("error"),"Erro diário em itens");
}
function render(data) {
  currentView=data; $("content").hidden=false;
  table("metrics",data.metrics,metricColumns);
  table("points",data.points,[["candidate","Modelo"],["cd_setor","Setor"],["relative_day","Dia do ciclo"],["date","Data"],["actual_share","Share real (0–1)"],["share_pred","Share previsto (0–1)"],["actual","Real (itens)"],["items_pred","Previsto (itens)"],["error","Erro (itens)"]]);
  $("point-count").textContent=`Mostrando ${data.points.length} de ${fmt(data.n_points,0)} linhas. Baixe o CSV para consultar todas.`;
  const best=data.metrics[0], cards=[["Menor Share MAE",fmt(best.share_mae_pp)+" p.p.",best.candidate],["Setores no recorte",fmt(data.n_sectors,0),"Pontos comuns aos modelos"],["Curvas avaliadas",fmt(best.n_sector_cycles,0),"Por modelo"],["Dias avaliados",fmt(best.n_days,0),"Por modelo"]];
  $("kpis").replaceChildren(...cards.map(([label,value,note])=>{const card=document.createElement("div");card.className="kpi";for(const [tag,text] of [["small",label],["strong",value],["small",note]]){const node=document.createElement(tag);node.textContent=text;card.append(node);}return card;}));
  $("slice").textContent=`Ciclo ${$("cycle").value} · origem ${$("origin").value}`;
  plots(data);
}
async function update() {
  const serial=++ticket; $("content").hidden=true; message("Calculando o recorte…");
  try { const params=query(), data=await request("/api/view",params); if(serial!==ticket)return;
    render(data); $("download").href="/api/export?"+new URLSearchParams(params); message("Recorte atualizado. Passe o mouse nos pontos para consultar os valores.");
  } catch(error) { if(serial===ticket)message(error.message,true); }
}
function origins() { choices("origin",runOptions.origins_by_cycle[$("cycle").value]||[]); }
async function loadRun() {
  const serial=++ticket; $("content").hidden=true; $("filters").hidden=true;
  try { const result=await request("/api/options",{run:$("run").value}); if(serial!==ticket)return; runOptions=result; choices("model",runOptions.models,"Todos os modelos"); choices("cycle",runOptions.cycles); origins();
    $("sectors").replaceChildren(...runOptions.sectors.map(s=>new Option(s,s))); $("sector").value="";
    $("manifest").textContent=JSON.stringify(runOptions.manifest,null,2); table("official",runOptions.official_metrics,metricColumns);
    $("config").href="/api/config?"+new URLSearchParams({run:$("run").value}); $("filters").hidden=false; await update();
  } catch(error) { if(serial===ticket)message(error.message,true); }
}
async function reload() {
  ++ticket; $("content").hidden=true; $("filters").hidden=true;
  try { const runs=await request("/api/runs"); choices("run",runs); $("empty").hidden=!!runs.length;
    if(runs.length)await loadRun(); else message("Nenhuma pasta de shape encontrada. Você pode configurar outra raiz usando --runs.");
  } catch(error) { message("Servidor indisponível. Execute: .venv\\Scripts\\python.exe -m shape_dashboard.serve",true); }
}
$("reload").addEventListener("click",reload); $("run").addEventListener("change",loadRun);
$("cycle").addEventListener("change",()=>{origins();update();});
for(const id of ["model","origin"]) $(id).addEventListener("change",update);
let debounce; $("sector").addEventListener("input",()=>{clearTimeout(debounce);debounce=setTimeout(update,300);});
$("unit").addEventListener("change",()=>{if(currentView)plots(currentView);});
$("clear").addEventListener("click",()=>{$("sector").value="";$("model").value="";update();});
reload();
