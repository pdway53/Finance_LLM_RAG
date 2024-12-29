# Finance_LLM_RAG
finance and insurance LLM QA with RAG

### AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用
QA問答與RAG 資料來源: https://tbrain.trendmicro.com.tw/Competitions/Details/37

## Retrieval Result
Total : 150 questions, 1035 finance-related files and 643 insurance-related files</br>
Training Data Retrival Accuracy:</br>
- Insurance Accuaray:45/50</br>
- Finance Accuaray:44/50</br>
- FAQ : 48/50<br>

## LLM Answer Result
LLM 回答參考 QA_answer.json<br>
```shell
        {
            "Q": "匯款銀行及中間行所收取之相關費用由誰負擔?",
            "LLM": "匯款人負擔。"
        },
        {
            "Q": "本公司應在效力停止日前多少天以書面通知要保人？",
            "LLM": "本公司應在效力停止日前三個月以書面通知要保人。"
        },
        {
            "Q": "「完全失能保險金」的受益人是否可以指定或變更?",
            "LLM": "「完全失能保險金」的受益人，除非要保人依照契約規定指定或變更，否則不得指定或變更。"
        },
        {
            "Q": "本契約內容的變更應經由誰同意並批註？",
            "LLM": "要保人与保险公司双方书面或其他约定方式同意，并由保险公司批注或发给批注书。"
        },
        {
            "Q": "「基本保額」是指什麼？",
            "LLM": "「基本保額」是指對應之保單價值準備金與累計增額繳清保險基本保額對應之保單價值準備金加總之值。"
        }
```

## 檔案說明
文問範例 : questions_example.json</br>
Retrieval Ground True : ground_truths_example.json </br>
Retrieval 預測結果 : pred_retrieve_list.json</br>
QA 問題結果: QA_answer.json</br>

## LLM retrieval 
執行File retrieval
```shell
python main.py --question_path dataset/preliminary/questions_example.json --source_path ../datasets/reference --output_path ../datasets/dataset/preliminary/pred_retrieve_test.json --qa_gt_path ../datasets/dataset/preliminary/ground_truths_example.json --rerank_with_LLM 1 --test_traindata 1
```

### LLM QA Inference
執行LLM QA 回答
```shell
python evaluate.py --question_path questions_example.json --source_path PDF_FOLDER_PATH --output_path rag_retrieve_predict.json --qa_gt_path ground_truths_example.json  --llm_answer 1 --output_qa_path ../datasets/dataset/preliminary/QA_ans.json
```
