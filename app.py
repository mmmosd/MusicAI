import gradio as gr
import model

def func(input):
    return input

model.printTFVersion()

demo = gr.Interface(func, gr.inputs.Slider(0, 100), "text")
demo.launch()