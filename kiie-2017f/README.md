
### running step:
#### 1. 데이터셋 준비
{1} = '2007to2011' or '2012to2016'; {2} = '07to11' or '12to16'  
1-a1. [import_raw_dataset.py](./import_raw_dataset.py)  
USPTO DB로부터 데이터셋 불러옴
  - input: /data_json/uspto_db_connection_info.json
  - output: /data/data_{1}.csv  

1-b1. [set_subclass_Svc.py](./set_subclass_Svc.py)  
분석대상 Service subclass 정의(추출)
  - output: /data_json/subclass_list_ini.json

1-b2. [set_subclass_Mfr.py](./set_subclass_Mfr.py)  
분석대상 Mfr. subclass 정의(추출)
  - input:
    - /data/data_2007to2011.csv
    - /data/data_2012to2016.csv
    - /data_json/subclass_list_ini.json
  - output: /data_json/subclass_list_res07to16.json

1-c1. [import_raw_citation.py](./import_raw_citation.py)  
USPTO DB 내에 존재하는 모든 인용정보 추출
  - input: /data_json/uspto_db_connection_info.json
  - output: ../df_raw_citation_v201706.pickle  

1-d1. [get_pair_pat_subcls.py](./get_pair_pat_subcls.py)  
특허문서로부터 subclass pair 집계, 특허/subclass 단위 출현횟수 집계
  - input:
    - /data/data_{1}.csv
    - /data_json/subclass_list_res07to16.json
  - output:
    - /res_tmp/pair_pat_subcls_{2}.csv
    - /res_tmp/gby_patent_{2}.csv
    - /res_tmp/gby_subclass_{2}.csv

1-e1. [get_citation_target_pat.py](./get_citation_target_pat.py)  
분석대상 특허의 전방/후방 인용정보 추출  
  - input:
    - ../df_raw_citation_v201706.pickle
    - /res_tmp/gby_patent_{2}.csv
  - output:
    - /data/citation_cited_{2}.csv
    - /data/citation_citing_{2}.csv
    - /res_tmp/gby_patent_{2}.csv

1-e2. [get_relation_pat_citation.py](./get_relation_pat_citation.py)  
기간별 동시인용(CC), 서지결합(BC) 매트릭스 생성  
*{a} = 'BC' or 'CC'; {a1} = 'cited' or 'citing'*  
  - input:
    - /data/citation_{a1}_{2}.csv
    - /res_tmp/gby_patent_{2}.csv
  - output: /res_tmp/relation_{a}_pat_{2}.csv

1-f1. [get_gephi_input.py](./get_gephi_input.py)  
Gephi(network) input file 생성
  - input:
    - /res_tmp/gby_subclass_07to11.csv
    - /res_tmp/gby_subclass_12to16.csv
    - /data_json/subclass_list_res07to16.json
  - output:
    - /gephi_input/input_data_edge_{2}.csv
    - /gephi_input/input_data_node.csv

#### 2. 모델링 데이터셋 변수 생성    
- output(공통):
  - /dataset_model/data_subclass_{2}.csv
  - /dataset_model/data_subclass_{2}.pickle

2-1. [set_pair_dataset_mod0.py](./set_pair_dataset_mod0.py)  
기간별 동시분류(co-classification) 횟수 집계
  - input:
    - /res_tmp/pair_pat_subcls_{2}.csv
    - /res_tmp/gby_patent_{2}.csv
    - /res_tmp/gby_subclass_{2}.csv
  - output column: pair_id, **count_cooccur**

2-2. [set_pair_dataset_mod1.py](./set_pair_dataset_mod1.py)  
기간별 피인용(count_cited) 횟수 집계
  - input:
    - /dataset_model/data_pair_subclass_{2}.pickle
    - /res_tmp/gby_patent_{2}.csv
  - output column: pair_id, count_cooccur, **count_cited**

2-3a. [set_pair_dataset_mod2-1.py](./set_pair_dataset_mod2-1.py)  
기간별 동시인용(count_CC), 서지결합(count_BC) 횟수 집계
*{a} = 'BC' or 'CC'*
  - input:
    - /dataset_model/data_subclass_{2}.pickle
    - /res_tmp/gby_patent_{2}.csv
    - /res_tmp/relation_{a}_pat_{2}.csv
  - output column: pair_id, count_cooccur, count_cited, **count_CC, count_BC**

2-3b. [set_pair_dataset_mod2-2.py](./set_pair_dataset_mod2-2.py)  
기간별 동시인용(count_CC), 서지결합(count_BC) 횟수 ~~집계~~정규화  
  - input:
    - /dataset_model/data_subclass_{2}.pickle
    - /res_tmp/gby_subclass_{2}.csv
  - output column: pair_id, count_cooccur, count_cited, count_CC, count_BC, **norm_CC, norm_BC**

2-4. [set_pair_dataset_mod3.py](./set_pair_dataset_mod3.py)  
기간별 이웃노드기반 Typology 변수 추출
  - input:
    - /dataset_model/data_subclass_{2}.pickle
    - /gephi_input/input_data_edge_{2}.csv
  - output column: pair_id, count_cooccur, count_cited, count_CC, count_BC, norm_CC, norm_BC, **score_CN, score_JC, score_SI, score_SC, score_LHN, score_HP, score_HD, score_PA, score_AA, score_RA**

2-5. [set_pair_dataset_norm.py](./set_pair_dataset_norm.py)  
모델 학습/평가용 input data 정규화  
  - input: /dataset_model/data_subclass_{2}.pickle
  - output:
    - /dataset_model/feature_statistic_{2}.csv
    - /dataset_model/data_subclass_norm_{2}.csv
    - /dataset_model/data_subclass_norm_{2}.pickle

2*. [set_pair_output_result.py](./set_pair_output_result.py)  
모델 학습/평가용 output data 생성  
  - input(modeling):
    - /dataset_model/data_subclass_07to11.pickle
    - /gephi_input/input_data_edge_07to11.csv
    - /gephi_input/input_data_edge_12to16.csv
  - output(modeling):
    - /dataset_model/output_result_07to11.csv
    - /dataset_model/output_result_07to11.pickle
  - output column: **pair_id, true_07to11, true_12to16**
  - input(prediction):
    - /dataset_model/data_subclass_12to16.pickle
    - /gephi_input/input_data_edge_12to16.csv
  - output(prediction):
    - /dataset_model/output_result_12to16.csv
    - /dataset_model/output_result_12to16.pickle
  - output column: **pair_id, true_12to16**

#### 3. 모델링 및 예측, 해석
3-1. [modeling_training.py](./modeling_training.py)
  - input:
    - /dataset_model/data_subclass_norm_07to11.pickle
    - /dataset_model/output_result_07to11.pickle
  - output:
    - /clf_SVM.pkl
    - /clf_result_v2.csv

3-2. [modeling_prediction.py](./modeling_prediction.py)
  - input:
    - /clf_SVM.pkl
    - /dataset_model/data_subclass_norm_12to16.pickle
    - /dataset_model/output_result_12to16.pickle
  - output:
    - /dataset_model/output_result_12to16.pickle
  - output column: **pair_id, true_12to16, _prediction_**

3-3a. [get_gephi_input_pred.py](./get_gephi_input_pred.py)
  - input:
    - /dataset_model/output_result_12to16.pickle
    - /gephi_input/input_data_node.csv
    - /gephi_input/input_data_edge_12to16.csv
  - output: /gephi_input/input_data_edge_pred.csv

3-3b. [analyzing_subclass_pair.py](./analyzing_subclass_pair.py)
  - input:
    - /data_json/subclass_list_res07to16.json
    - /data_json/subclass_description.json   (?)
    - /dataset_model/output_result_07to11.pickle
    - /dataset_model/output_result_12to16.pickle
  - output:
    - /gephi_output/new_pair_in_12to16.csv
    - /gephi_output/new_pair_in_predict.csv
