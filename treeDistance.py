from xmldiff import main
import statistics
L=[]
S=0
for i in range(100):
    L.append(len(main.diff_files("exp/multilingual-lexical/english/predict_europarl_english/1_" + str(i) + ".xml", "exp/multilingual-lexical/french/predict_europarl_french/1_" + str(i) + ".xml")))
    S += len(main.diff_files("exp/multilingual-lexical/english/predict_europarl_english/1_" + str(i) + ".xml", "exp/multilingual-lexical/french/predict_europarl_french/1_" + str(i) + ".xml"))
    '''print("Number of differences :", L[i])'''
sortL = sorted(L)
print(sortL)
print(L)


