import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
st.wide = True
import base64

@st.cache_resource(ttl=24*60*60)
def adaboost_model(x_train, y_train, x_test, y_test, ada_estimators, ada_depth, learn_rate):
    base_estimator_ada = DecisionTreeClassifier(max_depth=ada_depth)
    ada = AdaBoostClassifier(base_estimator = base_estimator_ada, n_estimators = ada_estimators, learning_rate= learn_rate, random_state=0)
    ada.fit(x_train, y_train)
        
    ada_pred = ada.predict(x_test)
    ada_pred_proba = ada.predict_proba(x_test)[:,1]
    
    ada_accuracy = accuracy_score(y_test, ada_pred)
    ada_confusion_matrix = confusion_matrix(y_test, ada_pred)
    ada_classification_report = classification_report(y_test, ada_pred)
    ada_feature_importance = ada.feature_importances_

    return ada_pred, ada_pred_proba, ada_accuracy, ada_confusion_matrix, ada_classification_report, ada_feature_importance

@st.cache_resource(ttl=24*60*60)
def decision_model(x_train, y_train, x_test, y_test, criterion, dtc_depth, dtc_split, dtc_leaf):
    dtc = DecisionTreeClassifier(criterion = criterion, max_depth=dtc_depth, min_samples_split=dtc_split, min_samples_leaf=dtc_leaf, random_state=0)
    dtc.fit(x_train, y_train)
    
    dtc_pred = dtc.predict(x_test)
    dtc_pred_proba = dtc.predict_proba(x_test)[:,1]
    
    dtc_accuracy = accuracy_score(y_test, dtc_pred)
    dtc_confusion_matrix = confusion_matrix(y_test, dtc_pred)
    dtc_classification_report = classification_report(y_test, dtc_pred)
    dtc_feature_importance = dtc.feature_importances_
    dtc_tree = tree.export_graphviz(dtc, out_file=None, feature_names=x_train.columns, filled=True, rounded=True, special_characters=True)

    return dtc_pred, dtc_pred_proba, dtc_accuracy, dtc_confusion_matrix, dtc_classification_report, dtc_feature_importance, dtc_tree

@st.cache_resource(ttl=24*60*60)
def tree_visualization(dtc_tree):
    plt.figure(figsize=(25,25))
    return st.graphviz_chart(dtc_tree)

def images(fath):
    col1, col2, col3 = st.columns([0.4,1,0.4])

    with col1:
        st.write(' ')

    with col2:
        st.image(fath, use_column_width = True)

    with col3:
        st.write(' ')

    return None


def supervised_learning():
    st.title("Supervised Learning")

def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">다운로드</a>'
    return href


with st.spinner('Updating Report...'):
    try:
        x_train = st.session_state['x_train']
        x_test = st.session_state['x_test']
        y_train = st.session_state['y_train'] 
        y_test = st.session_state['y_test']
        normal_value = st.session_state['normal_value']
        abnormal_value = st.session_state['abnormal_value']  

    except:
        st.error('main페이지에서 데이터를 전송해주세요.')
        st.stop()

    
    numeric_columns = x_train.select_dtypes(include=['int64', 'float64']).columns
    numeric_train = x_train[numeric_columns]
    numeric_test = x_test[numeric_columns]
    
    st.title('Supervised Learning')
    st.markdown('----')
    st.subheader('추천 데이터 셋 : Ford A, B')

    st.markdown('----')
    st.markdown('데이터 스케일')
    if st.checkbox('StandardScaler'):
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_train)
        numeric_test = scaler.transform(numeric_test)
        st.write('스케일 완료')
        numeric_test = pd.DataFrame(numeric_test, columns=numeric_columns)
        numeric_train = pd.DataFrame(numeric_train, columns=numeric_columns)


    st.markdown('----')
    st.markdown('## 1. Decision Tree Classifier')
    with st.expander('Decision Tree 설명'):
        st.markdown('Decision Tree는 :red[분할 기준 선택 -> 데이터 분할 -> 재귀적 분할 -> 종료 조건] 순으로 진행되는 분류 알고리즘입니다.')
        st.markdown('### 1. 분할 기준 선택 :')
        st.markdown('Decison Tree는 입력 특서 중 가장 유용한 특성과 그 특성의 분할 기준을 선택합니다.')
        st.markdown('각 특성의 중요도를 평가하고, 분할 기준에 따른 불순도(impurity)를 계산합니다.')
        st.markdown('불순도는 특정 그룹 내의 클래스의 다양성을 측정하는 것으로, 불순도가 낮을수록 분할이 잘 된다고 할 수 있습니다.')
        
        images('./images/dt_1.png')

        st.markdown('위의 사진이 불순도를 나타낸 그림입니다.')

        st.markdown('### 2. 데이터 분할 :')
        st.markdown('선택된 분할 기준을 사용하여 두 개 이상의 하위 그룹으로 데이터를 분할합니다.')
        st.markdown('여기서 Infomation gain의 개념이 들어가게 됩니다.')
        st.markdown('Infomation gain은 엔트로피의 변화량을 측정하여 얼마나 많은 정보를 얻을 수 있는지 판단합니다.')
        st.markdown('Infomation gain이 높을수록 분할이 잘 된 것으로 판단합니다.')
        st.markdown('따라서, Decision Tree의 데이터 분할은 Infomation gain이 높은 방향으로 진행이 됩니다.')
        st.markdown('이때 Infomation gain이 최대화 되는 지점을 찾는 과정은 greedy search 방식을 사용합니다.')
        st.markdown('물론 현재를 기준으로 하기 때문에 지역적으로 최적의 선택을 하는 것일수도 있습니다.')
        st.markdown('하지만 많은 경우에 효과적인 트리 모델을 생성할 수 있게 되며, tree를 많이 만들어 앙상블하는 방식으로 이를 보완할 수 있습니다.')
        
        st.markdown('### 3. 재귀적 분할 :')
        st.markdown('데이터가 모두 분할될 때까지 분할 기준 선택과 데이터 분할을 반복합니다.')
        st.markdown('위에서 분할된 노드를 다시 새로운 노드로 표현하며, 계속해서 분할을 진행합니다.')

        st.markdown('### 4. 종료 조건 :')
        st.markdown('데이터가 모두 분할되었거나, 더 이상 분할할 수 없는 경우 종료합니다.')
        st.markdown('이때 종료 조건은 다음과 같습니다.')
        st.markdown('1) Max Depth (최대 트리 깊이): 트리의 깊이가 지정된 최대 깊이에 도달하는 경우')
        st.markdown('2) Min Samples (최소 데이터 개수) : 노드에 포함된 데이터의 개수가 지정된 최소 개수 미만이 되는 경우')
        st.markdown('3) Min Impurity Decrease (최소 불순도) : 분할로 인한 불순도 감소가 지정된 최소 불순도 감소 미만이 되는 경우')
        st.markdown('4) Pure Node (노드에 속한 클래스 동일) : 분할된 노드에 속한 데이터가 모두 동일한 클래스인 경우')
        st.markdown(' ')
        images('./images/dt_2.png')
        st.markdown('위의 사진은 Decision Tree의 예시입니다.')

    st.markdown('### Hyperparameter Tuning')

    dtc_criterion = st.selectbox('criterion', ['gini', 'entropy'], help = '노드 분할의 기준으로 사용되는 지표, gini = 지니 불순도, entropy = 엔트로피')
    dtc_depth = st.number_input('max_depth (5 ~ 7)', min_value=5, max_value=7, value=5, step=1, format=None, key=None, help = '트리의 최대 깊이')
    dtc_split = st.number_input('min_samples_split (2 ~ 15)', min_value=2, max_value=15, value=2, step=1, format=None, key=None, help = '노드를 분할하기 위한 최소한의 샘플 데이터 수')
    dtc_leaf = st.number_input('min_samples_leaf (1 ~ 15)', min_value=1, max_value=15, value=1, step=1, format=None, key=None, help = '리프 노드가 되기 위한 최소한의 샘플 데이터 수')

    if st.checkbox('Decision Tree 실행'):
        dtc_pred, dtc_pred_proba, dtc_accuracy, dtc_confusion_matrix, dtc_classification_report, dtc_feature_importance, dtc_tree = decision_model(numeric_train, y_train, numeric_test, y_test, dtc_criterion, dtc_depth, dtc_split, dtc_leaf)
    
        if st.checkbox('Decision Tree Feature Importance 확인'):
            fig_dtc_importance = px.bar(x=numeric_train.columns, y=dtc_feature_importance)
            st.plotly_chart(fig_dtc_importance)
        
        # if st.checkbox('Decision Tree 결과 시각화'):
        #     fig_dtc = px.scatter(x = x_test.iloc[:,0], y = x_test.iloc[:,1], color = dtc_pred)
        #     st.plotly_chart(fig_dtc)

        if st.checkbox('Decision Tree 결과 확인'):
            st.write('Accuracy : ', dtc_accuracy)
            st.write('Confusion Matrix : ', dtc_confusion_matrix)
            st.write('Classification Report : ')
            st.code('Model Report:\n  ' + dtc_classification_report)
        
            dtc_fpr, dtc_tpr, dtc_thresholds = roc_curve(y_test, dtc_pred_proba)
            st.write('AUC: ', auc(dtc_fpr, dtc_tpr))
            fig_dtc_roc = px.area(x=dtc_fpr, y=dtc_tpr, title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=600, height=600)
            fig_dtc_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_dtc_roc.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_dtc_roc.update_xaxes(constrain='domain')
            st.plotly_chart(fig_dtc_roc)



        if st.checkbox('Decision Tree 시각화'):
            plt.figure(figsize=(25,25))
            tree_visualization(dtc_tree)

        if st.button('Result Download'):
            total_result_data = pd.DataFrame({'real':y_test, 'pred':dtc_pred})
            total_result_data.to_csv('total_result_data.csv', index=False)

            csv_link = download_csv(total_result_data, 'Decision_Tree_Result.csv')
            st.markdown(csv_link, unsafe_allow_html=True)

    st.markdown('----')
    st.markdown('## 2. AdaBoost(Adaptive Boost) Classifier')

    
    with st.expander('AdaBoost(Adaptive Boost) 설명'):
        st.markdown('AdaBoost는 Decision Tree가 Ensemble된 모형입니다.')
        st.markdown('Ensemble은 여러 개의 모형을 결합하여 더 나은 성능을 내는 모형을 만드는 방법입니다.')
        st.markdown('Ensemble은 아래 추가적으로 공부할 내용에서 다루고, 현재에는 Boosting에 대해서만 알아보겠습니다.')
        st.markdown('### 1. 초기 가중치 할당 :')
        st.markdown('각 데이터 포인트에 대해 초기 가중치가 할당됩니다. 이 가중치는 모두 동일하게 초기화됩니다.')
        st.markdown(' ')
        st.markdown('### 2. 기본 분류기(weak learner) 학습 :')
        st.markdown('첫 번째 분류기가 가중치가 적용된 데이터셋을 이용하여 학습합니다.')
        st.markdown('기본 분류기는 보통 Decision Tree입니다. 초기 가중치를 기반으로 학습 데이터 셋에 대해 분류 모델이 생성됩니다.')
        st.markdown(' ')
        st.markdown('### 3. 분류 오차 계산 :')
        st.markdown('첫 번째 분류기를 사용하여 학습 데이터셋을 분류하고 실제 클래스와 비교하여 오차를 계산하게 됩니다.')
        st.markdown('이때 분류 ㅇ차는 각 데이터 포인트의 가중치에 의해 가중평균으로 계산합니다.')
        st.markdown(' ')
        st.markdown('### 4. 가중치 업데이트 :')
        st.markdown('분류 오차를기반으로 데이터 포인트의 가중치를 업데이트 합니다.')
        st.markdown('분류 오차가 높은 데이터 포인트는 가중치가 증가하고, 분류 오차가 낮은 데이터 포인트는 가중치가 감소합니다.')
        st.markdown('이때, 분류 오차가 0인 데이터 포인트는 가중치가 변하지 않습니다.')
        images('./images/Boosting2.png')    
        st.markdown('위의 사진은 가중치를 업데이트하는 전반적인 사진입니다.')
        st.markdown(' ')
        st.markdown('### 5. 새로운 분류기 학습 :')
        st.markdown('위의 과정을 통해 업데이트 된 가중치를 기반으로 다음 기본 분류기를 학습합니다.')
        st.markdown('이전 단계에서 업데이트 된 가중치를 기반으로 새로운 분류기가 학습됩니다.')
        st.markdown(' ')
        st.markdown('### 6. 앙상블 모델 생성 :')
        st.markdown('위의 과정을 반복하여 모든 기본 분류기가 학습되면, 모든 분류기의 예측을 합산하여 가장 높은 예측을 가진 클래스를 선택합니다.')
        st.markdown('이 과정을 통해 여러 개의 Decision Tree가 학습됩니다.')
        st.markdown(' ')
        st.markdown('### 7. 앙상블 모델 예측 :')
        st.markdown('학습된 기본 분류기들을 앙상블하여 하나의 예측 모델을 생성합니다.')
        st.markdown('가중치는 각 분류기의 학습 성능을 기반으로 결정합니다.')
        st.markdown(' ')
        st.markdown('### 8. 최종 모델 평가 :')
        st.markdown('앙상블된 모델의 성능을 평가합니다.')
        images('./images/Boosting3.png')
        st.markdown('최종적인 과정은 위와 같습니다')


    #st.markdown('----')
    st.markdown('### Hyperparameter Tuning')
    ada_estimators = st.number_input('n_estimators (100 ~ 500)', min_value=100, max_value=500, value=200, step=100, format=None, key=None,
                                     help = '약한 학습기의 개수, 트리의 수를 결정, 수가 증가할 수록 모델의 복잡도가 증가하고 과적합 가능성이 증가')
    ada_depth = st.number_input('max_depth (3 ~ 7)', min_value=3, max_value=7, value=3, step=1, format=None, key=None, 
                                help = '약한 학습기의 최대 깊이를 제한, 일반화 성능 향상에 기여')
    ada_learn = st.number_input('learning_rate (0.1 ~ 1.0)', min_value=0.1, max_value=1.0, value=0.1, step=0.1, format=None, key=None,
                                help = '학습률, 각각의 약한 학습기가 학습에 기여하는 정도, 수가 증가할 수록 복잡도가 증가하고 과적합 가능성이 증가')
    

        

    if st.checkbox('AdaBoost 실행'):
        ada_pred, ada_pred_proba, ada_accuracy, ada_confusion_matrix, ada_classification_report, ada_feature_importance = adaboost_model(numeric_train, y_train, numeric_test, y_test, ada_estimators, ada_depth, ada_learn)    
        
        if st.checkbox('Feature Importance 확인'):
            fig_ada_importance = px.bar(x=numeric_train.columns, y=ada_feature_importance)
            st.plotly_chart(fig_ada_importance)
        
        # if st.checkbox('AdaBoost 결과 시각화'):
        #     fig_ada = px.scatter(x = x_test.iloc[:,0], y = x_test.iloc[:,1], color = ada_pred)
        #     st.plotly_chart(fig_ada)

        if st.checkbox('AdaBoost 결과 확인'):
            st.write('Accuracy: ', ada_accuracy)
            st.write('Confusion Matrix: ', ada_confusion_matrix)
            st.write('Classification Report: ')
            st.code('Model Report:\n  ' + ada_classification_report)

            ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada_pred_proba)
            st.write('AUC: ', auc(ada_fpr, ada_tpr))
            fig_ada_roc = px.area(x=ada_fpr, y=ada_tpr, title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=600, height=600)
            fig_ada_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_ada_roc.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_ada_roc.update_xaxes(constrain='domain')
            st.plotly_chart(fig_ada_roc)

        if st.button('Result Download'):
            total_result_data = pd.DataFrame({'real':y_test, 'pred':ada_pred})
            total_result_data.to_csv('total_result_data.csv', index=False)

            csv_link = download_csv(total_result_data, 'AdaBoost_Result.csv')
            st.markdown(csv_link, unsafe_allow_html=True)

    st.markdown('---')
    with st.expander('추가적으로 공부할 Decision Tree 관련 내용'):
        st.markdown('### Ensemble이란?')
        st.markdown('Ensemble은 여러 개의 분류기를 생성하고, 예측 값을 결합함으로써 보다 정확한 최종 예측 값을 도출하는 기법을 말한다.')
        st.markdown('여러개의 모델을 사용하여 하나의 모델을 만드는 과정인데, 이런 과정에서 크게 3가지로 배깅(bagging), 부스팅(boosting), 스태킹(stacking)이 있다.')
        st.markdown('그리고 보팅(voting)도 존재한다.')
        
        st.markdown('우선 앙상블 기법에 대해 알아보기 전에 bais(편향), variance(분산)에 대한 개념이 있어야 한다.')
        st.markdown('bais는 예측값과 실제값의 차이로, 예측값이 실제값과 얼마나 다른지를 나타내는 지표이다.')
        images('./images/bias.png')
        st.markdown('bias의 수식을 표현하면 위와 같다.')
        st.markdown(' ')
        st.markdown('variance는 예측값들이 서로 얼마나 퍼져있는지를 나타내는 지표이다.')
        images('./images/variance.jpg')
        st.markdown('variance의 수식을 표현하면 위와 같다.')
        st.markdown(' ')
        st.markdown('그러면 bais와 variance의 관계에 대해서 확인해보겠다.')
        images('./images/bias_variance2.png')
        st.markdown('그림으로 나타내면 위와 같다.')
        st.markdown('우리의 목표는 low varince, low bias를 가지는 모델을 만드는 것이다.')
        st.markdown('하지만 이 두가지는 trade-off 관계에 있고 train data에 두가지를 모두 만족시킨다는 것은 overfitting이 되는 것에 가깝다.')
        st.markdown('그렇기 때문에 우리는 각 모델의 특성에 따라 bias와 variance를 조절해야 한다.')
        st.markdown('high variance, low bias인 모델은 variance를 낮추기 위해 boosting을 사용하는 것이 좋다.')
        st.markdown('반면에 high bias, low variance인 모델은 bias를 낮추기 위해 bagging을 사용하는 것이 좋다.')
        st.markdown('물론 모든 경우에 대해 이렇게 말할 수는 없지만, 대부분의 경우에 해당한다.')
        images('./images/bias_variance.png')
        st.markdown('위의 그림을 보면, 우리는 결국 두가지의 케이스를 가운데 있는 medium bais, medium variance로 만들어야 한다.')
        st.markdown('이를 위해 우리는 앙상블 기법을 사용한다.')
        st.markdown(' ')
        images('./images/bias_variance3.png')
        st.markdown('bias와 variance의 관계는 결국 상충적이기 때문에 위와 같은 그래프가 나오게 된다.')
        st.markdown('이는 train, test data에서 많이 보았던 그래프 모양과 비슷하므로, 쉽게 이해할 수 있을 것으로 생각된다.')
        
        st.markdown('----')
        st.markdown('## Ensemble기법')
        st.markdown(' ')
        
        st.markdown('### 1. 보팅 (Voting) :')
        st.markdown('보팅은 각 개별모델이 투표를 통해 최종 예측 결과를 결정하는 방식이다.')
        st.markdown('hard voting과 soft voting이 있다.')
        st.markdown('hard voting은 다수결 원칙과 비슷하게 다수의 분류기가 결정한 예측값을 최종 보팅 결과값으로 선정하는 방식이다.')
        st.markdown('soft voting은 분류기들의 레이블 값 결정 확률을 모두 더하고, 이를 평균(가중평균) 해서 확률이 가장 높은 레이블 값을 최종 보팅 결과값으로 선정하는 방식이다.')
        images('./images/voting_1.png')
        st.markdown('위의 그림은 voting의 예시이다.')
        st.markdown(' ')
        
        st.markdown('### 2. 배깅 (Bagging) :')
        st.markdown('배깅은 원본 데이터에서 여러 번 샘플링(Boostrap)하여 여러 모델을 학습시키고, 이들의 평균치를 사용해 예측하는 방식이다.')
        st.markdown('이러한 과정에서 각 모델의 노이즈와 오차를 상호 상쇄시키는 효과가 발생한다고 할 수 있다.')
        st.markdown('이 방식은 모델의 Variance를 줄이고, Overfitting을 방지하며 모델의 일반화 성능을 향상 시킨다.')
        images('./images/bagging.png')
        st.markdown('위의 그림은 배깅의 예시이다.')
        st.markdown(' ')
        
        st.markdown('### 3. 부스팅 (Boosting) :')
        st.markdown('Boosting의 대표 알고리즘 중 하나인 위에서 소개한 :red[Adaboost]에 있는 설명을 참고 하는 것이 좋습니다.')
        st.markdown('부스팅은 배깅과 유사하게 여러 개의 분류기를 사용하지만, 순차적으로 학습을 수행한다는 차이점이 있다.')
        st.markdown('이러한 방식은 모델의 bais를 줄이는 것에 초점을 맞추는 것이다.')
        st.markdown('')
        st.markdown(' ')
        images('./images/boosting.png')
        st.markdown('위의 그림은 부스팅의 예시이다.')
        st.markdown(' ')
        st.markdown('### 4. 스태킹 (Stacking) :')
        st.markdown('스태킹은 여러 개의 다른 모델의 예측 결과값을 다시 학습 데이터로 만들어서 다른 모델(메타 모델)로 재학습시켜 결과를 예측하는 방식이다.')
        st.markdown('이러한 과정에서 각 모델의 장점을 사용하며, 단점을 보완할 수 있다.')
        st.markdown('다양한 모델의 장점을 사용한다는 측면에서 모델의 다양성이 확보될 때 더 좋은 성능을 기대할 수 있다.')
        st.markdown('최종 아웃풋을 사용하기 위해 메타 모델을 사용해야 하는데, 메타 모델에서 각 기본 모델의 예측이 최종 예측에 어떻게 기여하는지를 학습한다.')
        st.markdown(' ')
        images('./images/stacking.png')
        st.markdown('위의 그림은 스태킹의 예시이다.')
        
        st.markdown(' ')
        
        st.markdown('### 총 정리 :')
        st.markdown(':red[배깅]은 모델의 :red[분산]을 줄이며, :red[과적합을 방지]하는 것에 효과적')
        st.markdown(':red[부스팅]은 모델의 :red[편향]을 줄이며, :red[성능을 향상]시키는 것에 효과적')
        st.markdown(':red[스태킹]은 :red[여러 모델의 장점]을 모두 활용하므로, 모델의 :red[다양성을 확보]할 수록 더 좋은 성능을 기대할 수 있다.')