import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io
import numpy as np

def plot_confusion(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    return fig

def plot_roc(y_true, y_proba, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
    ax.legend()
    return fig

def model_comparison_bar(cv_scores):
    names = list(cv_scores.keys())
    vals = [cv_scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=vals, y=names, orient='h', ax=ax)
    ax.set_xlabel('CV Accuracy'); ax.set_title('Model CV Comparison')
    return fig

def save_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_report(pdf_path, cv_scores, top_models, X_hold, y_hold, final_blend, stacking, metrics):
    # choose final using metrics
    chosen_model = stacking if metrics.get('stack_acc',0)>=metrics.get('blend_acc',0) else final_blend
    chosen_label = 'Stacking' if metrics.get('stack_acc',0)>=metrics.get('blend_acc',0) else 'Blending'

    # get probabilities if possible
    if hasattr(chosen_model, 'predict_proba'):
        y_proba = chosen_model.predict_proba(X_hold)[:,1]
    else:
        # fallback - use decision_function if present
        if hasattr(chosen_model,'decision_function'):
            y_proba = chosen_model.decision_function(X_hold)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max()-y_proba.min()+1e-9)
        else:
            y_proba = np.zeros(len(y_hold))

    y_pred = chosen_model.predict(X_hold)
    acc = accuracy_score(y_hold, y_pred)
    prec = precision_score(y_hold, y_pred, zero_division=0)
    rec = recall_score(y_hold, y_pred, zero_division=0)
    f1 = f1_score(y_hold, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_hold, y_proba) if y_proba is not None else 0.0

    # build plots
    fig1 = model_comparison_bar(cv_scores)
    fig2 = plot_confusion(y_hold, y_pred, title=f'Confusion Matrix - {chosen_label}')
    fig3 = plot_roc(y_hold, y_proba, title=f'ROC - {chosen_label}')

    # write PDF using reportlab adding images of plots
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-50, "Model Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height-70, f"Chosen model: {chosen_label}")
    c.drawString(50, height-90, f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC AUC: {roc_auc:.4f}")

    # add CV scores table
    ypos = height-120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, ypos, "Model CV Scores")
    ypos -= 14
    c.setFont("Helvetica", 10)
    for k,v in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
        c.drawString(60, ypos, f"{k}: {v:.4f}")
        ypos -= 12

    # add images
    x_img = 50
    y_img = ypos - 20
    for fig in (fig1, fig2, fig3):
        imgbuf = save_fig_to_bytes(fig)
        img = ImageReader(imgbuf)
        c.drawImage(img, x_img, y_img-200, width=500, height=200)
        y_img -= 220
        if y_img < 100:
            c.showPage()
            y_img = height-80

    c.save()
