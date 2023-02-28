import time
from datetime import datetime

from chatbot.generator.util import Generator
from chatbot.pipeline.data_pipeline import DataPipeline
from chatbot.retriever.elastic_retriever import ElasticRetriever
from twitter.tweet_pipeline import TwitterPipeline
from utils.classes import BotReply, UserTweet
from database.mongodb import MongoDB
from spam_filter.spam_filter import SpamFilter

from omegaconf import OmegaConf
from pytz import timezone

# fmt: off
special_tokens = ["BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해" ] #TO-Do
# fmt: on


def main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator): # 🔺 db
    today = datetime.now(timezone("Asia/Seoul")).strftime("%m%d")

    # twitter api에서 메시지 불러오기
    new_tweets = twitter_pipeline.get_mentions()
    if len(new_tweets) == 0:
        # 새 메시지가 없으면
        time.sleep(60.0)
    else:
        for tweet in reversed(new_tweets):
            time_log = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            user_message = tweet.message.lower()

            # 스팸 필터링
            is_spam = spam_filter.sentences_predict(user_message)  # 1이면 스팸, 0이면 아님
            if is_spam:
                my_reply = reply_to_spam = "...."
                twitter_pipeline.reply_tweet(tweet=tweet, reply=reply_to_spam)
                score = 0.0
            else:
                # 리트리버
                retrieved = elastic_retriever.return_answer(user_message)
                if retrieved.query is not None:
                    my_reply = data_pipeline.correct_grammar(retrieved)
                    score = retrieved.bm25_score
                else:
                    # 생성모델
                    my_reply = generator.get_answer(user_message, 1, 256)
                    # 후처리
                    my_reply = data_pipeline.postprocess(my_reply, tweet.user_screen_name)
                    score = 0.0
                # twitter로 보내기
                twitter_pipeline.reply_tweet(tweet=tweet, reply=my_reply)
                # twitter 좋아요
                twitter_pipeline.like_tweet(tweet)

            # logging
            record = BotReply(
                tweet=tweet,
                reply=my_reply,
                score=score,
                is_spam=bool(is_spam),
                time=time_log,
            ).__dict__
            print(record)
            #db.insert_one(record) # 🔺

    return main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator) # db


if __name__ == "__main__":
    config = OmegaConf.load("./utils/base_config.yaml")

    # init modules
    spam_filter = SpamFilter()
    twitter_pipeline = TwitterPipeline(FILE_NAME="./twitter/last_seen_id.txt", bot_username="wjlee_nlp")
    data_pipeline = DataPipeline(log_dir="log", special_tokens=special_tokens)
    elastic_retriever = ElasticRetriever()
    generator = Generator(config)
    # db = MongoDB()

    main(spam_filter, twitter_pipeline, data_pipeline, elastic_retriever, generator) # 🔺 db
