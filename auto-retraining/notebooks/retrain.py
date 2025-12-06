import os
import time
import schedule
import subprocess

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def retrain_sentiment():
    p = os.path.join(BASE, "sentiment-service", "notebooks", "train_sentiment.py")
    subprocess.run(["python", p], check=False)

def main():
    schedule.every().week.do(retrain_sentiment)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
