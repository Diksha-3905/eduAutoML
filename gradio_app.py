"""
eduAutoML — Gradio Dashboard
Drag-and-drop dataset → EDA → Train → Explain
"""

import os
import tempfile

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from eduautoml.automl import AutoML

_automl_instance = None
_df_cache = None
_output_dir = tempfile.mkdtemp(prefix="eduautoml_")


def load_csv(file):
    global _df_cache
    if file is None:
        return gr.update(choices=[], value=None), "❌ No file uploaded.", None
    df = pd.read_csv(file.name)
    _df_cache = df
    cols = list(df.columns)
    preview = df.head(8).to_html(index=False, border=0)
    info = f"✅ Loaded **{df.shape[0]} rows × {df.shape[1]} columns**\n\nColumns: {', '.join(cols)}"
    return gr.update(choices=cols, value=cols[-1]), info, preview


def run_eda(target_col, learning_mode):
    global _df_cache
    if _df_cache is None:
        return "❌ Please upload a CSV first.", None, None, None, None
    aml = AutoML(learning_mode=learning_mode)
    stats, paths = aml.eda_report(_df_cache, target_col, output_dir=_output_dir)
    msg = (
        f"### 📊 EDA Summary\n"
        f"- **Shape:** {stats['shape'][0]} rows, {stats['shape'][1]} columns\n"
        f"- **Missing values:** {sum(1 for v in stats['missing'].values() if v > 0)} columns affected\n"
    )
    if "class_balance" in stats:
        cls = stats["class_balance"]
        msg += f"- **Classes:** {', '.join(str(k) for k in cls.keys())}\n"
        sizes = list(cls.values())
        if len(sizes) > 1 and max(sizes) / (min(sizes) + 1e-9) > 3:
            msg += "- ⚠️  **Class imbalance detected** — consider oversampling (SMOTE)\n"
    imgs = [paths[i] if i < len(paths) else None for i in range(4)]
    return msg, *imgs


def run_training(target_col, learning_mode):
    global _automl_instance, _df_cache
    if _df_cache is None:
        return "❌ Upload a CSV first.", None, None
    _automl_instance = AutoML(learning_mode=learning_mode)
    _automl_instance.fit(_df_cache, target_col)
    comp_df, comp_img = _automl_instance.compare_models(output_dir=_output_dir)
    best = _automl_instance.best_model_name
    score = comp_df.iloc[0, 1]
    task = _automl_instance.task
    result_md = (
        f"### 🏆 Training Complete\n"
        f"- **Task:** {task.capitalize()}\n"
        f"- **Best Model:** `{best}`\n"
        f"- **Score:** `{score:.4f}` ({'Accuracy' if task == 'classification' else 'R²'})\n\n"
        f"#### 📋 Model Comparison\n{comp_df.to_markdown(index=False)}"
    )
    return result_md, comp_img, comp_df


def run_explain():
    global _automl_instance
    if _automl_instance is None:
        return "❌ Train a model first.", None, None, None
    paths = _automl_instance.explain(output_dir=_output_dir)
    path_map = {os.path.basename(p): p for p in paths}
    msg = f"### 🔬 Explainability Report\nModel: `{_automl_instance.best_model_name}`\n\nCharts show which features matter most for predictions.\nSHAP values show direction and magnitude of each feature's impact."
    fi  = path_map.get("feature_importance.png")
    shp = path_map.get("shap_summary.png")
    cm  = path_map.get("confusion_matrix.png")
    return msg, fi, shp, cm


def save_model():
    global _automl_instance
    if _automl_instance is None:
        return None, "❌ Train a model first."
    path = os.path.join(_output_dir, "automl_model.pkl")
    _automl_instance.save(path)
    return path, f"✅ Model saved as `automl_model.pkl`"


def build_app():
    css = """
    .title-block{text-align:center;padding:20px 0 10px}
    .title-block h1{font-size:2.2rem;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    """

    with gr.Blocks(css=css, title="eduAutoML") as demo:
        gr.HTML("""
        <div class="title-block">
          <h1>🎓 eduAutoML</h1>
          <p style="color:#555">Beginner-friendly · Explainable · Educational AutoML</p>
        </div>
        """)

        # Shared state
        target_dropdown = gr.Dropdown(label="🎯 Target Column", choices=[], interactive=True, visible=False)
        learning_toggle = gr.Checkbox(label="🧠 Learning Mode", value=False, visible=False)

        with gr.Tab("📂 1. Upload Dataset"):
            file_input = gr.File(label="Drag & Drop your CSV here", file_types=[".csv"])
            load_btn   = gr.Button("Load Dataset", variant="primary")
            target_vis = gr.Dropdown(label="🎯 Select Target Column", choices=[], interactive=True)
            learning_vis = gr.Checkbox(label="🧠 Enable Learning Mode (explains each step)", value=False)
            upload_info = gr.Markdown("Upload a CSV to begin.")
            dataset_preview = gr.HTML()

            def load_and_sync(file):
                r1, r2, r3 = load_csv(file)
                return r1, r1, r2, r3

            load_btn.click(
                load_csv,
                inputs=[file_input],
                outputs=[target_vis, upload_info, dataset_preview],
            )
            target_vis.change(lambda v: v, inputs=target_vis, outputs=target_dropdown)
            learning_vis.change(lambda v: v, inputs=learning_vis, outputs=learning_toggle)

        with gr.Tab("📊 2. Auto EDA"):
            eda_target   = gr.Dropdown(label="Target Column", choices=[], interactive=True)
            eda_learning = gr.Checkbox(label="Learning Mode", value=False)
            eda_btn      = gr.Button("▶ Run EDA Report", variant="primary")
            eda_summary  = gr.Markdown()
            with gr.Row():
                eda_img1 = gr.Image(label="Missing Values", type="filepath")
                eda_img2 = gr.Image(label="Target Distribution", type="filepath")
            with gr.Row():
                eda_img3 = gr.Image(label="Correlation Heatmap", type="filepath")
                eda_img4 = gr.Image(label="Feature Distributions", type="filepath")
            eda_btn.click(run_eda, inputs=[eda_target, eda_learning],
                          outputs=[eda_summary, eda_img1, eda_img2, eda_img3, eda_img4])
            # Sync from upload tab
            load_btn.click(load_csv, inputs=[file_input], outputs=[eda_target, gr.Markdown(visible=False), gr.HTML(visible=False)])

        with gr.Tab("🏋️ 3. Train Models"):
            train_target   = gr.Dropdown(label="Target Column", choices=[], interactive=True)
            train_learning = gr.Checkbox(label="Learning Mode", value=False)
            train_btn      = gr.Button("🚀 Start Training", variant="primary")
            train_result   = gr.Markdown()
            comp_img       = gr.Image(label="Model Comparison Chart", type="filepath")
            comp_table     = gr.Dataframe(label="Detailed Results")
            train_btn.click(run_training, inputs=[train_target, train_learning],
                            outputs=[train_result, comp_img, comp_table])
            load_btn.click(load_csv, inputs=[file_input], outputs=[train_target, gr.Markdown(visible=False), gr.HTML(visible=False)])

        with gr.Tab("🔬 4. Explain Model"):
            explain_btn = gr.Button("Generate Explanations", variant="primary")
            explain_md  = gr.Markdown()
            with gr.Row():
                exp_fi   = gr.Image(label="Feature Importance", type="filepath")
                exp_shap = gr.Image(label="SHAP Summary", type="filepath")
            exp_cm = gr.Image(label="Confusion Matrix", type="filepath")
            explain_btn.click(run_explain, inputs=[], outputs=[explain_md, exp_fi, exp_shap, exp_cm])

        with gr.Tab("💾 5. Save Model"):
            save_btn  = gr.Button("Download Trained Model (.pkl)", variant="primary")
            save_msg  = gr.Markdown()
            save_file = gr.File(label="Download your model")
            save_btn.click(save_model, inputs=[], outputs=[save_file, save_msg])

        gr.HTML("""
        <div style="text-align:center;padding:20px;color:#888;font-size:0.85rem">
          Built with ❤️ by Diksha Wagh &nbsp;|&nbsp;
          <a href="https://github.com/Diksha-3905/eduAutoML" target="_blank">GitHub</a>
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
