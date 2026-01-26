''' 
high level implementation of a decoder from BLIP 

Can be combined with CLIP to make an encoder & decoder model based on Language modeling loss 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 1. Causal Self-Attention (Masked)
        self.masked_self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        # 2. Cross-Attention (To "see" the image/context)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        # 3. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)




    def forward(self, x, context_features, causal_mask):
        # x: Current text sequence [Batch, Seq_Len, D]
        # context_features: Image features from Encoder [Batch, Num_Patches, Dimension(width)]
        # causual mask: (Seq_len, seq_len) 


        # Self-Attention with Causal Mask
        # Nx seq_len 
        attn_out, _ = self.masked_self_attn(x, x, x, attn_mask=causal_mask)

        # apply layer norm and residual connection for numerical stability (no vanishing and exploding gradients) 
        x = self.ln1(x + attn_out)

        # Cross-Attention: Query comes from text (x), Keys/Values from Context

        # cross attention (Nxseq_len x D) 
        cross_out, _ = self.cross_attn(query=x, key=context_features, value=context_features)
        
        x = self.ln2(x + cross_out)

        # Feed Forward back to the original dimension 
        x = self.ln3(x + self.ffn(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        # embed the tokens 
        self.embedding = nn.Embedding(vocab_size, d_model)

        # go through the decoder layers (transformer internal layers) 
        self.layers = nn.ModuleList([SimpleDecoderLayer(d_model) for _ in range(6)])
        
        # transforms (N x seq_len x D) where each row is a token 
        # map the hidden dimension back into vocab_size, which becomes the logits of each corresponding word 
        # N x seq_len x vocab size 

        ''''        
        '''
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, context_features):
        ''' 
        tokens: 
        '''
        # tokens: [N, sequence_len] 
        x = self.embedding(tokens)
        
        # Create a causal mask (triangular) so token i cannot see token i+1
        seq_len = x.size(1)
        
        ''' 
        batch size = 1 sentence, 2 tokens(words), d_embed= 3  
       input: Q, K= 
                [1,1,1]
                    [2,2,2]
        W_q = W_k = 3 x2 
        casual mask= 
            [0,1]
            [0,0]
        
            Q @K_T = 2x2 

        apply attention_mask to the output of Q * K_T 


        attention with causual mask: 
            first word only sees it self 
            second word only sees first and second 
            ... 

                after softmax is applied, only the the words that should be seen have a score  

        this is why the casual mask is seq_len x seq_len, where seq_len = row # of Q

        After Q*K_T, the matrix size becomes seq_len x seq_len 
        then apply the mask as addition, turning vlaues that are 1 into -inf

            During the training process, the input is the entire sentence 
        but because of the attention mask, (remember the output of masked cross attention) is seq_len x seq_len
        each row of seq_len is a word 
        at each row, the amount of masking is different. First row: only has the first word, so in the output, the first word is the prediction of the second word 
            second row: sees the first two words, so it predicts the 3rd word and so on .... 


        '''
        # create upper triangular matrix where the diagonal and below are all 0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        for layer in self.layers:
            x = layer(x, context_features, causal_mask)
            

        # return the embedding 
        return self.head(x) # Returns logits for every token in vocabulary
    

def compute_lm_loss(logits, targets):
    """
    logits: [Batch, Seq_Len, Vocab_Size]
    targets: [Batch, Seq_Len]

    each row is a word, and because of seq_len, it outputs the entire sentence
    does prediction known as teacher forcing, which improves model training speed 

    by the way we have input, the prediction is for the second word and onwards (first row -> 2nd word)


    Training: 
        sentence to be learned: [The, cat, sits, on, mat, <END>]
        formatted input: [<START>, The, cat, sits, on, mat]
        seq_len = 6 
        labels: [The, cat, sits, on, mat, <END>]

    the first word <START> should predict "The", and so on 
    compute cross entropy on all words.
    
    during the data processing stage,
      we use a token to show the end of a single training sample and a token as the first input to the model to start predicting the actual training sample 
        
    """
    # 1. We must shift the targets. 
    # If input is [A, B, C], the targets for indices [0, 1] are [B, C].
    # PyTorch's CrossEntropy usually handles the 'shift' if you align your tensors correctly.
    
    # Flatten the batch and sequence dimensions for CrossEntropy
    # Shape: [Batch * Seq_Len, Vocab_Size]
    # flatten into one, as multiple labels to predict one label, which is exactly the setting of crossEntropy
    logits = logits.view(-1, logits.size(-1))
    
    # Shape: [Batch * Seq_Len]
    targets = targets.view(-1)
    
    # 2. Cross Entropy handles the Log-Softmax and NLLLoss automatically
    # ignore_index=0 handles padding tokens so they don't affect the gradient
    return F.cross_entropy(logits, targets, ignore_index=0)