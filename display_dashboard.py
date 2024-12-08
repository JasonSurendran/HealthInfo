import dash
from dash import html, dcc
import pandas as pd

# Load common words and their frequencies
def load_common_words(input_file):
    """
    Load common words and their frequencies from a text file into a DataFrame.
    """
    words = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, freq = line.strip().split("\t")
            words.append((word, int(freq)))
    return pd.DataFrame(words, columns=["Word", "Frequency"])


# Load document similarity results
def load_similarity_results(input_file):
    """
    Load document similarity results from a text file into a DataFrame.
    """
    similarities = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            doc_pair, similarity = line.strip().split(": ")
            similarities.append((doc_pair.strip(), float(similarity)))
    return pd.DataFrame(similarities, columns=["Document Pair", "Cosine Similarity"]).sort_values(
        by="Cosine Similarity", ascending=False
    )


# Load data
common_words_file = "output_files/common_words_filtered.txt"
similarity_file = "output_files/document_similarity.txt"

common_words_df = load_common_words(common_words_file)
similarity_df = load_similarity_results(similarity_file)

# Dash app setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Document Analysis Dashboard"

app.layout = html.Div([
    html.H1("Document Analysis Dashboard", style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Common Words", value="tab-1"),
        dcc.Tab(label="Document Similarity", value="tab-2"),
    ]),
    html.Div(id="tabs-content")
])


@app.callback(
    dash.dependencies.Output("tabs-content", "children"),
    [dash.dependencies.Input("tabs", "value")]
)
def render_tab_content(tab_name):
    if tab_name == "tab-1":
        return html.Div([
            html.H4("Top Common Words Across Documents"),
            dcc.Dropdown(
                id="num-words-dropdown",
                options=[{"label": f"Top {i}", "value": i} for i in range(5, 51, 5)],
                value=10,
                style={"width": "50%", "margin": "0 auto"}
            ),
            html.Div(id="common-words-output", style={"margin": "20px", "textAlign": "center"})
        ])
    elif tab_name == "tab-2":
        return html.Div([
            html.H4("Document Pair Similarity (Most Similar to Least Similar)"),
            html.Table(
                children=[
                    html.Thead(html.Tr([html.Th("Document Pair"), html.Th("Cosine Similarity")])),
                    html.Tbody([
                        html.Tr([html.Td(row["Document Pair"]), html.Td(f"{row['Cosine Similarity']:.4f}")])
                        for _, row in similarity_df.iterrows()
                    ])
                ],
                style={"margin": "0 auto", "width": "80%", "border": "1px solid black", "borderCollapse": "collapse"}
            )
        ])


@app.callback(
    dash.dependencies.Output("common-words-output", "children"),
    [dash.dependencies.Input("num-words-dropdown", "value")]
)
def update_common_words(num_words):
    # Select top `num_words` words
    top_words = common_words_df.head(num_words)
    return html.Table(
        children=[
            html.Thead(html.Tr([html.Th("Word"), html.Th("Frequency")])),
            html.Tbody([
                html.Tr([html.Td(row["Word"]), html.Td(row["Frequency"])])
                for _, row in top_words.iterrows()
            ])
        ],
        style={"margin": "0 auto", "width": "50%", "border": "1px solid black", "borderCollapse": "collapse"}
    )


def run():
    app.run_server(debug=True)
