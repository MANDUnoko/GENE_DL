[기획]
목적
비선형 유전자 상호작용 해석 가능성 확보
: 기존 AE  구조에서 latent vector로 어느 정도 유의미한 분류 결과 확보했으나 해석력 떨어지는 한계 존재 => Self-Attention 기반 Transformer 도입으로 해석 가능성 향상

장점:
Attention weight로 유전자 중요도 해석

분석 및 해석 계획
해석 기반 분석
FN 케이스 → Attention 기반 유전자 추적 (개별 환자군 해석)

기대 결과
AE 대비 해석 가능성이 높은 모델 구조 확보
FN 사례에 대해 생물학적 해석이 가능한 유전자 집합 도출

[진행]
AETransformerLite
: 소규모 샘플 수, 고차원 유전자 feature, 해석 가능성
이 세 조건을 모두 만족하는 모델이 필요했고,
그 해답으로 AE(차원 축소) + Self-Attention(해석 가능)을 결합한 경량 구조를 설계함.

1. Encoder (AE part)
self.encoder = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, latent_dim)
)
- 역할
고차원 유전자 입력 (18,000+) → **latent feature (8차원)**로 축소
중간층 64차원은 과도한 정보 압축 방지 역할

- 설계 이유
우리 데이터는 샘플 수가 작고 (n ≈ 60),
차원 수는 많아 (d ≈ 18,000) → 차원의 저주(curse of dimensionality) 극복 필요
단순 PCA와 달리, 비선형 변환으로 정보 손실 최소화
ReLU 활성화로 비선형성 보장

2. Embedding (reshape for Transformer input)
x = latent.unsqueeze(2)               # (B, latent_dim, 1)
x = self.embedding(x)                 # (B, latent_dim, tf_embed_dim)

- 역할
AE로 압축된 latent vector를 Transformer가 이해 가능한 시퀀스 형태로 변환
각 latent feature를 sequence의 token처럼 취급하여 attention 학습 가능

- 설계 이유
Transformer는 기본적으로 시계열/sequence 처리 구조
유전자 자체가 순서성을 갖지는 않지만, latent dimension 간의 상관관계를 token 관계로 해석
Linear(1 → tf_embed_dim)은 각 token(=latent dim)을 동일한 임베딩 공간으로 투사

3. Self-Attention (Transformer Core)
self.self_attn = nn.MultiheadAttention(embed_dim=tf_embed_dim, num_heads=1, ...)

- 역할
latent dimension 간 상호작용 학습: 어떤 latent가 다른 latent에 얼마나 영향 주는지 학습
attn_weights를 통해 해석 가능한 score matrix (L × L) 획득 가능

- 설계 이유
num_heads=1: 소규모 데이터에서는 attention head가 많을수록 noise 유입 ↑
Attention weight는 FN 환자의 feature 해석에 직접 활용 가능 → explainable AI

4. Global Average Pooling
x = attn_out.mean(dim=1)

- 역할
Transformer의 출력 시퀀스를 하나의 feature vector로 압축
전체 latent dimension에 대한 전반적인 요약 표현 생성

- 설계 이유
[CLS] token 도입 대신 mean pooling 사용 → 구조 간단 + 성능 유사
시퀀스 길이(latent_dim)가 작기 때문에 pooling 손실 없음

5. Classifier
self.ffn = nn.Sequential(
    nn.LayerNorm(tf_embed_dim),
    nn.Linear(tf_embed_dim, 1),
    nn.Sigmoid()
)

- 역할
Transformer의 pooled representation을 이진 분류로 변환
LayerNorm은 small batch 상황에서 안정성 향상

- 설계 이유
과적합 방지를 위해 단순 linear classifier 구성
Sigmoid 출력으로 binary classification → Stroke vs Normal

6. Attention 저장
self.attn_weights = attn_weights.detach().cpu()

- 역할
해석 가능성 확보의 핵심
각 latent 차원 간 상호 영향도를 저장하여, 이후 input gene → attention score 연결 가능

- 설계 이유
Transformer의 최대 장점 중 하나인 self-attention 시각화를 그대로 활용하기 위함
Encoder weight와 조합하면 유전자 수준의 영향력 도출 가능

=> 이 구조가 갖는 장점 (우리 데이터 특성 기준)

데이터 특성
 구조적 대응
샘플 수 적음 (n ≈ 60)
파라미터 수 최소화 (1-head, 1-layer TF, 작은 FFN)
차원 수 많음 (d ≈ 18,000)
AE로 차원 축소 후 latent 분석
해석 필요
attention weight + encoder 가중치 조합으로 gene-level 해석
FN case 해석
attn_weights + encoder.weight 이용하여 유전자 중요도 시각화 가능
불안정 학습 우려
dropout, LayerNorm, ReduceLROnPlateau 등 안정화 요소 포함


[학습 결과]
학습 손실은 0.61에서 0.18 수준까지 안정적으로 수렴하여, 모델이 효과적으로 학습되었음을 확인

테스트셋 기준 성능은 다음과 같음:
| Metric | Value |
|--------|-------|
| Accuracy | 0.8462 |
| Precision | 1.0000 |
| Recall (Sensitivity) | 0.7500 |
| Specificity | 1.0000 |
| F1 Score | 0.8571 |
| AUC | 0.9500 |

특히 AUC 0.95는 높은 분류 능력을 의미하며, 이는 소수 샘플에서도 학습이 효과적으로 이루어졌음을 보여줌

모델은 attn_weights (Transformer의 self-attention 출력)와 encoder.weight (AE의 gene-to-latent mapping)을 결합하여, FN 케이스에 대해 유전자 단위 중요도 분석이 가능

예시로 FN 판별된 환자 샘플 1번과 11번의 Top10 유전자 중요도는 아래와 같으며,
일부 유전자는 두 FN 케이스 모두에서 반복적으로 상위 랭크에 나타남:

| 유전자 | FN1 중요도 | FN11 중요도 |
|--------|-------------|--------------|
| ABCA1 | 0.0897 | 0.0868 |
| ABCB9 | 0.0850 | 0.0896 |
| AAA1 | 0.0858 | 0.0856 |
| AACS | 0.0848 | 0.0861 |
| ABCA3 | 0.0819 | 0.0838 |

→ 이전에 기능적 분석으로 도출한 결과와 비교
FN 케이스에서 공통적으로 높은 importance를 갖는 유전자들은ABC transporter 계열 (ABCA1, ABCA3, ABCA13, ABCB9)또는 세포 내 대사/수송 관련 유전자

비교 분석: 기존 기능적 해석 vs 모델 해석 (FN 기준)

항목
| 항목            | 기존 기능 해석 (DEG 분석 기반)      | FN 해석 유전자 기반 해석              |
|-----------------|--------------------------------------|----------------------------------------|
| 주요 기전     | 면역, 염증, 항체, 사이토카인         | 지질 수송, 막 단백질, 대사 조절         |
| 대표 유전자   | ARG1, CD6, MMP9, S100A12             | ABCA1, ABCB9, AACS, AATF               |
| 공통점        | 일부 세포막 수송 기전 포함            | ABCA1: known inflammation/lipid metabolism |
| 차이점        | **면역 반응 경로**에 집중됨          | **대사 및 막 수송 유전자** 쪽이 부각됨   |

FN이 발생한 원인에 대한 추론
모델이 면역 반응 관련 유전자 패턴을 학습하는 데에는 성공했으나, N 케이스는 이들과는 다른 생물학적 signature, 즉 대사 / 수송 중심 유전자들이 더 활성화된 케이스로 보임
즉:
이 환자들은 기존 뇌졸중 면역 signature가 강하게 드러나지 않았고 신 지질 대사, transporter 계열 유전자가 상대적으로 우세했기 때문에 델이 기존 학습 기반으로 Stroke로 인식하지 못함 → FN

주목할 점
기존 DEG 분석에서 놓치기 쉬운 "조금 다른 생물학적 subtype"일 가능성
FN이 단순히 "모델이 틀린 것"이 아니라, hidden subgroup을 보여주는 사례일 수 있음

[결론 및 의의]
본 모델은 소규모 고차원 유전자 발현 데이터에서도 다음과 같은 성과를 보임:

예측 성능 확보: 85% 이상의 정확도, 0.95의 AUC
해석 가능성 확보: FN 케이스 기반 유전자 중요도 추출 가능
경량화 구조 채택: 복잡한 Transformer 구조 없이도 self-attention 기반 해석 가능
AETransformerLite 기반 해석 결과는 기존 DEG 기반 면역 중심 해석과는 다른 생물학적 해석 축을 제시 (FN으로 분류된 샘플에서 반복적으로 등장한 ABCA1, ABCB9, ABCA3 등은 지질 수송 및 막 안정성에 관여하는 유전자로, 이는 클래식한 염증성 뇌졸중과는 다른 분자 서브타입의 가능성을 시사함. 즉, 이 결과는 모델 해석이 단순히 잘못된 예측을 설명하는 것이 아니라, 새로운 환자군 구분 혹은 잠재적 바이오마커 후보의 탐색에 활용될 수 있는 실마리를 제공)

따라서 본 구조는 특히 바이오마커 탐색, 소수 표본 기반 정밀의료 모델 개발, 의료 AI explainability 확장 분야에 활용될 수 있음

[한계와 확장성]
현재 모델은 latent-level attention 기반 해석만 가능하며, 추후 gene-wise Transformer 구조로 확장할 경우 직접적인 유전자 간 관계 해석도 가능할 것으로 보임(이를 실현하려면 더 많은 케이스가 필요함)

또한 추론 과정의 불확실성을 정량화하기 위한 Bayesian Transformer, 또는 GNN + Transformer 기반 연동도 고려할 수 있음
