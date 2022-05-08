## dl4h: Paper Reproduction Project

Original Paper:
>Ma, F., Wang, Y., Xiao, H. et al. Incorporating medical code descriptions for diagnosis prediction in healthcare. BMC Med Inform Decis Mak 19, 267 (2019). https://doi.org/10.1186/s12911-019-0961-2



### Questions
1. Text embeddings (fastText)
2. CNN implementation and embedding matrix
3. Use of embedding matrix ***E*** in enhanced models
4. Calculation of visit-level precision and code-level accuracy

### Setup
The code in this repository uses Python 3.8 and PyTorch 1.10.

It is highly recommended to create a Python virtual environment with Python 3.8 and then execute the following:

`pip install -r requirements.txt`

#### Text embeddings
Current implementation of word embeddings is in `EmbeddingCNN.ipynb`.

This takes the corpus of text from the full ICD9 Code descriptions, in lower case with punctuation and digits filtered out but no stop words removed (filtering is performed in `DataProcessing.ipynb`. The corpus is a text file with a row for each diagnosis code of very ICU visit. Embedding is performed with `fastText` using `skipgram` model with `dim=300`. 

Q: Is this the correct approach to acquiring embedding vectors for each word in the vocabulary?

###### *Python Implementation*
```python
ft_model = fasttext.train_unsupervised(OUTPUT_TEXT, model='skipgram', dim=300, minCount=1)
```
NOTE: `OUTPUT_TEXT` is a text file with a line of text for every diagnosis code in every ICU visit.

#### CNN implementation
Current implementation of CNN embedding matrix is also in `EmbeddingCNN.ipynb`.

This loads as input data all the unique ICD9 code descriptions at the lowest level (i.e. 4903 unique descriptions), and is trained to predict which code each description is by index value. During training the embedding matrix is passed through a dropout layer (0.5) and a final FC layer to get logits for CE Loss. After 100 training iterations the embedding matrix is returned after the Max Pooling layer to be used in the enhanced models.

Q: Is this the correct approach to training the embedding CNN?
###### *Python Implementation*
```python
class EmbeddingCNN(torch.nn.Module):

    def __init__(self, num_descriptions, embedding_dim, num_class, num_kernel, kernel_sizes):
        super().__init__()
        """
        Arguments:
            hidden_dim: the hidden dimension
            num_descriptions: number of icd9 descrptions
            embedding_dim: size of word embedding dim (from fastText)
            num_class: number of classes to predict
            num_kernel: number of filters for each kernel size
            kernel_sizes: list of sizes to iterate on
        """
        self.embed = nn.Embedding(num_descriptions, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernel, (K, embedding_dim)) for K in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_kernel, num_descriptions)

    def forward(self, x, masks):
        """
        Arguments:
            x: the input tensor of icd9 description of size (batch_size, max_num_words, word_embedding_dim) 
            masks: masks for the padded words of size (batch_size, max_num_words, word_embedding_dim)
        
        Outputs:
            logit: logits for cross entropy loss function to for training iterations
            embedding: embedding matrix of learned wieghts for icd9 descriptions
        """
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = mask_conv2d(x, masks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        embedding = torch.cat(x, 1)
        x = self.dropout(embedding)
        logit = self.fc(x)
        return logit, embedding

embedding_cnn = EmbeddingCNN(
    num_descriptions=len(codes), embedding_dim=300, num_class=len(codes),
    num_kernel=100, kernel_sizes=[2,3,4])
```
#### Use of ***E*** in enhanced models
Current implementation of EnhancedMLP is in `EnhancedMLP.ipynb`.

This loads the embedding layer ***E*** into FC layer of size (num_icd9codes, word_embedding_dim) *i.e. (4903, 300)*. The weights of this layer are assigned the values from embedding matrix ***E*** and `requires_grad` is set to `False` so that only the bias vector is learned. The input indices of ICD9 codes for each visit are converted to a multihot vector and passed through this layer, and then `tanh` is executed on the output.

Q: Is this the correct approach to obtaining the embedding vector **v**<sub>t</sub> as described for the enhanced method?

###### *Python Implementation*
```python
class EnhancedMLP(nn.Module):
    def __init__(self, num_codes, num_categories, embedding_matrix):
        super().__init__()
        """
        Arguments:
            num_codes: total number of diagnosis codes
            num_categories: number of higher level categories to predict
            embedding_matrix: learned embedding matrix of icd9 descriptions
        """
        self.embedding = nn.Linear(4903, 300)
        self.embedding.weight.data = embedding_matrix
        self.fc = nn.Linear(300, num_categories)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)
        Outputs:
            logits: logits of shape (batch_size, # diagnosis codes)
        """
        x = indices_to_multihot(x, masks, 4903)
        x = self.embedding(x)
        x = torch.tanh(x)
        logits = self.fc(x)
        return logits
    

# load the model here, set embedding_matrix to requires_grad=False (only learn bias vector)
enhanced_mlp = EnhancedMLP(num_codes = len(codes), num_categories=len(sub_categories), embedding_matrix=embedding_matrix)
for param in enhanced_mlp.named_parameters():
    if param[0] == "embedding.weight":
        param[1].requires_grad = False
enhanced_mlp
```
#### Calculation of precision/accuracy metrics
Current implementation is also in `EnhancedMLP.ipynb`.

This gets the logits from the model, then performs `softmax` along the last dim to get probabilty scores. For each visit the following visit info is predicted as the indices of the *top k* probabilities. The number of correct predictions per visit are counted as the number of indices in ***y*** that are in the *top k* predictions. Then visit-level precision is this count divided by `min(k, |y|)`. For code-level accuracy my current implementation is taking this same count of correct predictions and dividing by `|y|`. This seems incorrect but the values for baseline models seemed to most closely compare when calcluated like this. I initially had used ***k*** as the denominator since I understood this to be the total number of label predicitons (thinking all the *top k* are considered predictions).

Q: Is *visit-level precision@k* implemented correctly?

Q: How should I understand the total number of label predictions for *code-level accuracy@k*?

###### *Python Implementation*
```python
def eval_model(model, test_loader, k=15, n=-1):
    """
    Arguments:
        model: the EnhancedMLP model
        test_loader: validation dataloader
        
    Outputs:
        precision_k: visit-level precison@k
        accuracy_k: code-level accuracy@k
    """
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    all_precision = []
    all_accuracy = []
    
    model.eval()
    with torch.no_grad():
        for x, masks, y in test_loader:
            n_eval = y.shape[0] - 1 if n == -1 else n
            y_hat = model(x, masks)
            y_hat = F.softmax(y_hat, dim=-1)
            nz_rows, nz_cols = torch.nonzero(y, as_tuple=True)
            k_correct = 0
            total_precision = 0
            total_accuracy = 0
            for i in range(n_eval):
                visit_correct = 0
                y_true = nz_cols[nz_rows == i]
                _, y_pred = torch.topk(y_hat[i], k)

                for v in y_true:
                    if v in y_pred:
                        visit_correct += 1

                visit_precision = visit_correct / min(k, len(y_true))
                visit_accuracy = visit_correct / len(y_true)
                k_correct += visit_correct
                total_precision += visit_precision
                total_accuracy += visit_accuracy

            precision_k = total_precision / n_eval
            accuracy_k = total_accuracy / n_eval
            all_precision.append(precision_k)
            all_accuracy.append(accuracy_k)

    total_precision_k = np.mean(all_precision)
    total_accuracy_k = np.mean(all_accuracy)
    return total_precision_k, total_accuracy_k
```




UIUC CS598 Final Project
