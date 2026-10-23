"""Page-local blocks with explicit geometry; heuristic two-column ordering."""
import re
from collections import Counter
from statistics import median
import pymupdf

# 双栏页不能简单按纵坐标逐行读。先辨别左右栏，再以跨栏块分段，段内先左后右。
# 这是启发式排序，复杂表格、三栏、跨栏图片仍需要人工核对。
def ordered_blocks(blocks,width):
    middle=width/2
    left=[b for b in blocks if b['bbox'][2] <= middle+12]
    right=[b for b in blocks if b['bbox'][0] >= middle-12]
    is_columns=len(left)>=2 and len(right)>=2
    if not is_columns: return sorted(blocks,key=lambda b:(b['bbox'][1],b['bbox'][0])),False
    full=[b for b in blocks if b not in left and b not in right]
    remaining=list(blocks); ordered=[]
    for separator in sorted(full,key=lambda b:b['bbox'][1]):
        above=[b for b in remaining if b not in full and b['bbox'][3]<=separator['bbox'][1]+2]
        ordered.extend(sorted(above,key=lambda b:(0 if b in left else 1,b['bbox'][1])))
        for b in above: remaining.remove(b)
        ordered.append(separator);remaining.remove(separator)
    ordered.extend(sorted(remaining,key=lambda b:(0 if b in left else 1,b['bbox'][1])))
    return ordered,True

def parse_pdf(path):
    doc=pymupdf.open(path)
    try:
        if not doc.is_pdf or doc.needs_pass: raise ValueError('仅支持未加密的 PDF')
        if not 1<=len(doc)<=80: raise ValueError('PDF 页数须为 1–80 页')
        pages=[];margins=Counter();warnings=[]
        for page in doc:
            blocks=[]
            for b in page.get_text('dict',flags=pymupdf.TEXTFLAGS_TEXT)['blocks']:
                if b['type']!=0: continue
                lines=[''.join(s['text'] for s in line['spans']).strip() for line in b['lines']]
                content='\n'.join(x for x in lines if x).strip()
                if not content: continue
                sizes=[s['size'] for l in b['lines'] for s in l['spans']]
                # 从解析阶段就保留文件页码和 bbox，后面生成引用不需要让模型“猜页码”。
                block={'content':content,'bbox':list(b['bbox']),'font':max(sizes or [10]),'page':page.number+1}
                blocks.append(block)
                if b['bbox'][1]<42 or b['bbox'][3]>page.rect.height-35: margins[content]+=1
            pages.append((blocks,page.rect.width,page.rect.height))
        if sum(len(b['content']) for blocks,_,_ in pages for b in blocks)<50:
            raise ValueError('未提取到足够文本；扫描件需要 OCR，当前版本不支持')
        # 结合字号、位置和横向范围找标题，避免把页首版权说明或竖排 arXiv 标记当成题名。
        title_candidates=[b for b in pages[0][0] if 8<len(b['content'])<220 and
                          b['bbox'][2]-b['bbox'][0]>b['bbox'][3]-b['bbox'][1] and b['bbox'][1]<pages[0][2]*.55]
        title=max(title_candidates,key=lambda b:b['font'])['content'].replace('\n',' ') if title_candidates else None
        chunks=[];section='正文';ordinal=0
        for blocks,width,height in pages:
            # 只在页边缘过滤重复内容和页码，尽量避免删除正文中的重复术语。
            cleaned=[b for b in blocks if not ((b['bbox'][1]<42 or b['bbox'][3]>height-35) and
                       (margins[b['content']]>=2 or re.fullmatch(r'\d+',b['content'])))]
            body_size=median([b['font'] for b in cleaned] or [10])
            arranged,columns=ordered_blocks(cleaned,width)
            if columns: warnings.append(f'第 {blocks[0]["page"]} 页使用双栏启发式排序，请核对复杂跨栏内容。')
            if not cleaned: warnings.append('有页面未提取到文本，可能是图片页。')
            for b in arranged:
                content=b['content']
                if not title and b['page']==1: title=content.replace('\n',' ')[:240]
                if len(content)<120 and (b['font']>body_size*1.12 or re.match(r'^\d+(\.\d+)*\s+\w',content)):
                    section=content.replace('\n',' ')[:200]
                # Splits never cross page/block boundaries. Bbox highlights the originating block.
                # 页内、块内切分：最长 1200 字符，步长 1000，长块有 200 字符重叠。
                # 每个切片继承原始块 bbox，因此高亮可能大于摘录句子；不跨页保证页码可靠。
                for start in range(0,len(content),1000):
                    part=content[start:start+1200]
                    chunks.append({'page':b['page'],'ordinal':ordinal,'section':section,'bbox':b['bbox'],'content':part})
                    ordinal+=1
                    if start+1200>=len(content): break
                    if len(chunks)>5000: raise ValueError('文本块过多，超过本版限制')
        return {'title':title or 'Untitled paper','pages':len(doc),'warnings':warnings,
                'chunks':chunks,'parser':'pymupdf-blocks-v1'}
    finally: doc.close()
