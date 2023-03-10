### Usage

simple topic modeling utils based on BertTopic

```
from topic_generator import TopicGenerator

text = [YOUR TEXT EXCERPTS LIST]
embeddings = [YOUR CORRESPONDING VECTORS LIST]

topic_model = TopicGenerator(text, embeddings)
topic_model.get_total_topics()

topic_model.general_topics_df
```