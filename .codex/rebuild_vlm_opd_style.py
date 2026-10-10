from pathlib import Path
import subprocess, tempfile, re, html

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'vlm_工作整理_github_papers'

CSS='''<style>:root{--bg:#f8f9fa;--card:#fff;--text:#1a1a2e;--muted:#555;--accent:#2d6cdf;--light:#e8f0fe;--border:#e2e8f0}*{box-sizing:border-box}body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--text);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.7}.container{max-width:960px;margin:auto}.back{margin-bottom:1rem}.back a{color:var(--accent);text-decoration:none;font-weight:700}.paper-card{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 2px 8px #0000000d;padding:clamp(1.25rem,3vw,2rem)}h1{font-size:1.65rem;line-height:1.35;margin:0 0 .8rem}.meta-row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:1.2rem;font-size:.84rem}.tag,.meta-row a{padding:2px 9px;border-radius:99px;background:var(--light);color:var(--accent);text-decoration:none;font-weight:700}.summary-box{background:#f0f7ff;border-left:4px solid var(--accent);padding:.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:1.4rem}.section-title{font-size:1.12rem;margin:1.4rem 0 .6rem;border-bottom:1px dashed var(--border);padding-bottom:.3rem}.paper-card p,.paper-card li{font-size:.93rem;color:#333}.paper-card img{display:block;max-width:100%;height:auto;margin:1rem auto;border:1px solid var(--border);border-radius:8px}.paper-card table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.88rem}.paper-card th,.paper-card td{border:1px solid var(--border);padding:6px 9px}.paper-card th{background:var(--light)}.paper-card code{background:#f1f5f9;padding:2px 5px;border-radius:4px}.pros-cons{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.pros-cons>div{padding:.7rem .9rem;border-radius:8px}.pros{background:#f0fdf4;border-left:3px solid #22c55e}.cons{background:#fef2f2;border-left:3px solid #ef4444}@media(max-width:640px){.pros-cons{grid-template-columns:1fr}}</style>'''

def parse_sections(md):
    ms=list(re.finditer(r'^##\s+(.+?)\s*$',md,re.M)); out=[]
    for i,m in enumerate(ms):
        title=m.group(1).strip(); body=md[m.end():ms[i+1].start() if i+1<len(ms) else len(md)]
        out.append((title,body.strip()))
    return out

def pick(sections, terms):
    for title,body in sections:
        if any(t in title.lower() for t in terms) and '基本信息' not in title.lower(): return title,body
    return ('', '')

def render(markdown):
    markdown=markdown.replace('](assets/','](../vlm/assets/')
    with tempfile.NamedTemporaryFile('w',suffix='.md',encoding='utf-8',delete=False) as f:
        f.write(markdown); path=f.name
    try: return subprocess.check_output(['pandoc','-f','markdown+tex_math_dollars','-t','html',path],text=True,encoding='utf-8')
    finally: Path(path).unlink(missing_ok=True)

def clip(body, limit):
    body=body.strip()
    if len(body)<=limit: return body
    cut=body[:limit]
    # Keep complete paragraphs and preserve the first figure when it appears early.
    cut=cut.rsplit('\n\n',1)[0]
    return cut+'\n\n（本节其余细节见论文原始笔记。）'

def clean_heading(title):
    return re.sub(r'^\s*(?:\d+(?:\.\d+)*[.)]?\s*)+', '', title).strip()

def lead(body, limit=900):
    blocks=[]
    for block in re.split(r'\n\s*\n', body.strip()):
        stripped=block.strip()
        if not stripped or stripped.startswith(('#','![')):
            continue
        if stripped.startswith(('- ','* ','+ ')):
            continue
        blocks.append(stripped)
        if len(' '.join(blocks)) >= limit:
            break
    return clip('\n\n'.join(blocks), limit)

def first_image(body):
    match=re.search(r'!\[[^]]*\]\([^)]*\)', body)
    return render(match.group(0)) if match else ''

for src in (ROOT/'vlm').glob('*.md'):
    md=src.read_text(encoding='utf-8'); sections=parse_sections(md)
    title=re.search(r'^#\s+(.+)$',md,re.M).group(1).strip()
    info=pick(sections,['基本信息','论文基本信息'])[1]
    problem=pick(sections,['问题','动机','背景','讲什么','解决什么'])
    method=pick(sections,['方法','框架','思路','总览','核心机制','核心切入'])
    result=pick(sections,['实验','结果','结论','发现','效果','性能','评测','消融'])
    if not problem[1]: problem=('研究问题','论文围绕视觉语言模型中的视觉信息利用、推理质量或推理效率展开。')
    if not method[1]: method=('方法框架','作者提出针对上述问题的视觉语言模型训练或推理框架。')
    if not result[1]: result=('实验与结论','实验部分验证了方法在相关视觉语言任务上的有效性。')
    arxiv=re.search(r'ArXiv ID:\s*([0-9.]+)',md,re.I)
    abs_id=arxiv.group(1) if arxiv else ''
    info_html=render(info) if info else ''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},svg:{{fontCache:'global'}}}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>{CSS}</head><body><div class="container"><div class="back"><a href="../vlm_工作整理_github.html">← 返回 VLM 论文列表</a></div><article class="paper-card"><h1>{html.escape(title)}</h1><div class="meta-row"><span class="tag">VLM</span><span class="tag">视觉推理</span><span class="tag">多模态</span>{f'<a href="https://arxiv.org/abs/{abs_id}">arXiv {abs_id}</a>' if abs_id else ''}</div><div class="summary-box"><strong>核心问题：</strong>{render(lead(problem[1],450))}</div><h2 class="section-title">方法框架：{html.escape(clean_heading(method[0]))}</h2>{render(lead(method[1],900))}{first_image(method[1])}<h2 class="section-title">实验与核心结论：{html.escape(clean_heading(result[0]))}</h2>{render(lead(result[1],900))}{first_image(result[1])}<h2 class="section-title">论文信息</h2>{info_html}</article></div></body></html>'''
    (OUT/(src.stem+'.html')).write_text(page,encoding='utf-8')
print('rebuilt',len(list((ROOT/'vlm').glob('*.md'))),'VLM pages')
