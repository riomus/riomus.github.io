import { chromium } from 'playwright';
const BASE = 'http://localhost:4321';
const OUT = '.impeccable/review';
const pages = [['home','/'],['post','/2025/11/07/is-it-worth-2/'],['about','/about/'],
  ['writing','/writing/'],['software','/software/'],
  ['tag','/tags/dgx-spark/'],['notfound','/404.html']];
const sizes = [['desktop',1440,900],['mobile',390,844]];
const b = await chromium.launch();
for (const [sn,w,h] of sizes) {
  const ctx = await b.newContext({ viewport:{width:w,height:h}, deviceScaleFactor:2, reducedMotion:'reduce' });
  const p = await ctx.newPage();
  const errs=[]; p.on('console',m=>m.type()==='error'&&errs.push(m.text())); p.on('pageerror',e=>errs.push(String(e)));
  for (const [pn,path] of pages) {
    await p.goto(BASE+path,{waitUntil:'load'});
    await p.evaluate(() => document.fonts.ready);
    await p.evaluate(async () => {
      await new Promise(r => { let y=0; const s=setInterval(()=>{ window.scrollTo(0,y); y+=600;
        if (y > document.body.scrollHeight) { clearInterval(s); window.scrollTo(0,0); r(); } }, 25); });
      await Promise.all([...document.images].filter(i=>!i.complete).map(i=>new Promise(r=>{i.onload=i.onerror=r})));
    });
    await p.waitForTimeout(300);
    await p.screenshot({ path:`${OUT}/${sn}-${pn}.png`, fullPage:true });
    const of = await p.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth);
    const blank = await p.evaluate(()=>[...document.images].filter(i=>!i.complete||i.naturalWidth===0).length);
    console.log(`${sn}/${pn} overflow:${of}px unpainted:${blank}`);
  }
  if (errs.length) console.log(`${sn} ERRORS:`, errs.slice(0,5));
  await ctx.close();
}
await b.close();
