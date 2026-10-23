"""Small synthetic papers, explicitly NOT real research results."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

ROOT=Path(__file__).resolve().parents[1]

def paragraph(c,text,x,y,width=240):
    c.setFont('Helvetica',10)
    for line in simpleSplit(text,'Helvetica',10,width):c.drawString(x,y,line);y-=15
    return y-14

def heading(c,text,x,y):
    c.setFont('Helvetica-Bold',12);c.drawString(x,y,text);return y-22

def make(path,title,variant=False):
    c=canvas.Canvas(str(path),pagesize=(612,792));c.setTitle(title)
    c.setFont('Helvetica',8);c.drawString(42,765,'SYNTHETIC TEST FIXTURE - NOT A REAL RESEARCH PAPER')
    c.setFont('Helvetica-Bold',19);c.drawString(42,729,title)
    y=heading(c,'1 Introduction',42,686)
    y=paragraph(c,'This document is a controlled fixture for evaluating PDF evidence retrieval. All datasets and experimental results below are fictional.',42,y)
    y=heading(c,'2 Method',42,y)
    y=paragraph(c,'Our method uses a sparse keyword retriever and a local reranker. Evidence passages are retained with exact page references.' if not variant else 'Our method uses dense retrieval and a contrastive encoder. The system retains source coordinates for every passage.',42,y)
    y=heading(c,'3 Datasets',330,686)
    y=paragraph(c,'We evaluate on the fictional CedarQA dataset, containing 2400 training questions and 600 test questions.' if not variant else 'We evaluate on the fictional MapleQA dataset, containing 1800 training questions and 400 test questions.',330,y)
    y=heading(c,'4 Evaluation metrics',330,y)
    paragraph(c,'The evaluation metric is exact-match accuracy. The reported accuracy is 82.4 percent on CedarQA.' if not variant else 'The evaluation metric is macro F1. The reported macro F1 score is 0.78 on MapleQA.',330,y)
    c.setFont('Helvetica',9);c.drawString(300,24,'1');c.showPage()
    c.setFont('Helvetica',8);c.drawString(42,765,'SYNTHETIC TEST FIXTURE - NOT A REAL RESEARCH PAPER')
    y=heading(c,'5 Ablation study',42,719)
    y=paragraph(c,'Removing the reranker reduces exact-match accuracy from 82.4 to 76.1 percent. This ablation suggests the reranker improves retrieval quality on this fictional benchmark.' if not variant else 'Removing contrastive training reduces macro F1 from 0.78 to 0.69. The ablation measures the impact of contrastive supervision.',42,y)
    y=heading(c,'6 Limitations',330,719)
    y=paragraph(c,'The method was evaluated only on English text. Scanned PDFs and tables were not evaluated. Generalization to other languages remains unknown.' if not variant else 'The method requires GPU training and was tested on one fictional dataset. Performance on long documents was not evaluated.',330,y)
    y=heading(c,'7 Notes',42,510)
    paragraph(c,'No hardware latency, power consumption, clinical outcome, or deployment cost was measured. This fixture is intended for software testing only.',42,y,width=520)
    c.setFont('Helvetica',9);c.drawString(300,24,'2');c.save()

if __name__=='__main__':
    folder=ROOT/'fixtures';folder.mkdir(exist_ok=True)
    make(folder/'cedar.pdf','Cedar: Evidence Retrieval Study')
    make(folder/'maple.pdf','Maple: Contrastive Retrieval',True)
    print('Created two explicitly synthetic fixtures')
