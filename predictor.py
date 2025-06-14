def draw_to_vector(draw, total=43):
    vec = [0]*total
    for n in draw:
        vec[n - 1] = 1
    return vec

def predict_next(model, last_draw):
    vec = draw_to_vector(last_draw)
    probs = model.predict_proba([vec])[0]
    sorted_probs = sorted([(i+1, p) for i, p in enumerate(probs)], key=lambda x: -x[1])
    return [n for n, _ in sorted_probs[:6]]
