import torch
import gluonnlp as nlp
import numpy as np

from .BERTDataset import BERTDataset
from .model.BERTClassifier import BERTClassifier

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# parameter 
max_len = 128
batch_size = 32

# device 
device = torch.device('cpu')

#BERT 모델, Vocabulary 불러오기 
bertmodel, vocab = get_pytorch_kobert_model()

## 학습 모델 불러오기
PATH = './base/views/sentimentAnalysis/model/model_naver_11st.pt'
model = BERTClassifier(bert=bertmodel)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# 토큰화 
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class review_SA():

    def predict(predict_sentence):

        data = [predict_sentence, 0]
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_loader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                return np.argmax(logits)
