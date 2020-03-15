# MAML

We investigate applying MAML to boost performance on binary content moderation tasks, like sentiment analysis and insincere question detection, in low-resource contexts. We used the MAML algorithm, implemented in PyTorch, to train a model whose internal representation is amenable to a variety of content moderation tasks with minimal finetuning. Our distribution of content moderation tasks comprised 8 tasks, including sentiment analysis and insincere question detection, each with a separate dataset. We pre-trained our model using MAML and compared its ability to adapt to perform well on unseen binary content moderation tasks to that of a model pre-trained using traditional transfer learning approaches and a model trained from scratch. Empirically, we found that MAML did not improve adaptation performance significantly over traditional approaches to transfer learning. However, we hope that, with additional improvements to MAML and fewer memory and computational resource limitations, MAML can be applied to train robust and adaptive large-scale content moderation systems in low-resource contexts.
