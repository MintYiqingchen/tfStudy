import cntk as C
def display_model(model):
    from IPython.display import SVG, display
    svg = C.logging.graph.plot(model, "tmp1.svg")
    display(SVG(filename="tmp1.svg"))