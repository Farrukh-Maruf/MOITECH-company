import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
from PIL import Image
#from yolov8 import YOLO  # Assuming yolov8 is the module where your YOLO code resides
from ultralytics import YOLO
# Initialize the YOLO model
infer = YOLO("/best.pt")

# Create Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Button('Detect Objects', id='detect-button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='output-image-upload'),
    html.Div(id='output-prediction')
])


# Callback to display uploaded image
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(content, filename):
    if content is not None:
        return html.Div([
            html.H5(filename),
            html.Img(src=content, style={'height': '50%', 'width': '50%'})
        ])


# Callback to perform object detection
@app.callback(Output('output-prediction', 'children'),
              [Input('detect-button', 'n_clicks')],
              [State('upload-image', 'contents')])
def detect_objects(n_clicks, image_content):
    if n_clicks == 0:
        raise PreventUpdate

    if image_content is not None:
        # Convert the uploaded image to PIL format
        _, content_string = image_content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Perform object detection
        predictions = infer.predict(image)

        # Convert predictions to text
        predictions_text = "\n".join(predictions)

        return html.Div([
            html.Hr(),
            html.H5('Object Detection Results:'),
            html.Pre(predictions_text)
        ])
    else:
        return html.Div([
            html.Hr(),
            html.H5('No image uploaded.')
        ])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=2000, threaded=True)
