from flask import *
import pickle
import numpy as np

# 수정된 파일을 자동으로 reload한다
app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
)

# models라는 global dictionary 선언한다
models = {}

def init():
    with open("./models/classification.pkl", "rb") as f:
        models["classification"] = pickle.load(f)

# 서버에 접속하면 index.html 파일 내용을 렌더링한다
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predic")
def predic():
    result = {}
    result["category"] = ["정치", "경제", "사회", "생활/문화", "세계", "IT/과학"]

    model = models["classification"]

    sentence = request.values.get("sentence")

    result["sentence"] = sentence

    result["result"] = list(np.round_(model.predict_proba([sentence])[0] * 100, 2))

    return jsonify(result)


init()
app.run()
