# 사용자 설정: 결과 조회만 원하면 기본값으로 모두 실행하세요.
# ACTION="results"는 재학습하지 않습니다. 선택 기능의 목적은 COLAB.md에 설명되어 있습니다.
ACTION = "results"  # @param ["results", "predict", "agent", "evaluate", "train"]
FOLDER = "/content/drive/MyDrive/PINN"  # @param {type:"string"}
ZIP_FILENAME = "Online_PINN_PDE_Framework_V3_Complete_Results.zip"  # @param {type:"string"}
EQUATION = "poisson2d"  # @param {type:"string"}
ARM = "baseline"  # @param ["baseline", "loss_sa", "ff", "ff_loss_sa", "ff_curriculum"]
SEED = 3234  # @param {type:"integer"}
# POINTS의 좌표 순서와 범위는 선택한 방정식에 맞춰 입력하세요.
POINTS = [[0.25, 0.65]]
QUESTION = "Poisson에서 방법별 물리장 오차를 비교해줘"
USE_LLM = False  # @param {type:"boolean"}
DENSE_RAG = True  # @param {type:"boolean"}
EXECUTION = "new"  # @param ["new", "resume"]
RESUME_ZIP_FILENAME = ""  # @param {type:"string"}
# TRAIN_OVERRIDES: 학습 설정 변경안. cap은 누적 Adam 횟수, field_target은 비율(0.10=10%).
# TRAINING_ENABLED=False에서는 설정만 표시합니다. 기본값 전체는 실행 전에 출력됩니다.
# 예: {"arms": ["ff"], "cap": 100000, "base_lr": 0.002, "field_target": 0.10}
TRAIN_OVERRIDES = {}
TRAINING_ENABLED = False  # @param {type:"boolean"}
# True: evaluate에서 일회성 새 학습/재개 검사 실행. 단순 결과 조회에는 영향이 없습니다.
CHECK_TRAINING = False  # @param {type:"boolean"}
