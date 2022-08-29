from review_SA import review_SA

while True:
    sentence = input("긍부정을 판단할 리뷰를 입력해주세요 : ")
    if sentence == "0":
      print(">> 긍부정 판단을 종료합니다!\n")
      break
    print(review_SA.predict(sentence))
    print("\n")
