
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
import yolov9

def yolov9_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):

    # Load the model
    # model_path = download_models(model_id)
    model = yolov9.load(model_id)
    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    # Perform inference
    results = model(img_path, size=image_size)
    # Optionally, show detection bounding boxes on image
    output = results.render()
    return output[0]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                model_path = gr.Dropdown(
                    label="Model",
                    choices=[
                        "best.pt",
                    ],
                    value="./best.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                )
                yolov9_infer = gr.Button(value="Inference")

            with gr.Column():
                output_numpy = gr.Image(type="numpy",label="Output")

        yolov9_infer.click(
            fn=yolov9_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
        )
        



gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv9: Detect Void Space in Retail Shelf
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)
