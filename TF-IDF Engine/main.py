from __future__ import unicode_literals
import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import bisect 
from hazm import *
from parsivar import FindStems
import pickle


# stop words are from hazm github
stop_words=['و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای', 'آن', 'یک', 'خود', 'تا', 'کرد', 'بر', 'هم', 'نیز', 'گفت', 'می‌شود', 'وی', 'شد', 'دارد', 'ما', 'اما', 'یا', 'شده', 'باید', 'هر', 'آنها', 'بود', 'او', 'دیگر', 'دو', 'مورد', 'می‌کند', 'شود', 'کند', 'وجود', 'بین', 'پیش', 'شده_است', 'پس', 'نظر', 'اگر', 'همه', 'یکی', 'حال', 'هستند', 'من', 'کنند', 'نیست', 'باشد', 'چه', 'بی', 'می', 'بخش', 'می‌کنند', 'همین', 'افزود', 'هایی', 'دارند', 'راه', 'همچنین', 'روی', 'داد', 'بیشتر', 'بسیار', 'سه', 'داشت', 'چند', 'سوی', 'تنها', 'هیچ', 'میان', 'اینکه', 'شدن', 'بعد', 'جدید', 'ولی', 'حتی', 'کردن', 'برخی', 'کردند', 'می‌دهد', 'اول', 'نه', 'کرده_است', 'نسبت', 'بیش', 'شما', 'چنین', 'طور', 'افراد', 'تمام', 'درباره', 'بار', 'بسیاری', 'می‌تواند', 'کرده', 'چون', 'ندارد', 'دوم', 'بزرگ', 'طی', 'حدود', 'همان', 'بدون', 'البته', 'آنان', 'می‌گوید', 'دیگری', 'خواهد_شد', 'کنیم', 'قابل', 'یعنی', 'رشد', 'می‌توان', 'وارد', 'کل', 'ویژه', 'قبل', 'براساس', 'نیاز', 'گذاری', 'هنوز', 'لازم', 'سازی', 'بوده_است', 'چرا', 'می‌شوند', 'وقتی', 'گرفت', 'کم', 'جای', 'حالی', 'تغییر', 'پیدا', 'اکنون', 'تحت', 'باعث', 'مدت', 'فقط', 'زیادی', 'تعداد', 'آیا', 'بیان', 'رو', 'شدند', 'عدم', 'کرده_اند', 'بودن', 'نوع', 'بلکه', 'جاری', 'دهد', 'برابر', 'مهم', 'بوده', 'اخیر', 'مربوط', 'امر', 'زیر', 'گیری', 'شاید', 'خصوص', 'آقای', 'اثر', 'کننده', 'بودند', 'فکر', 'کنار', 'اولین', 'سوم', 'سایر', 'کنید', 'ضمن', 'مانند', 'باز', 'می‌گیرد', 'ممکن', 'حل', 'دارای', 'پی', 'مثل', 'می‌رسد', 'اجرا', 'دور', 'منظور', 'کسی', 'موجب', 'طول', 'امکان', 'آنچه', 'تعیین', 'گفته', 'شوند', 'جمع', 'خیلی', 'علاوه', 'گونه', 'تاکنون', 'رسید', 'ساله', 'گرفته', 'شده_اند', 'علت', 'چهار', 'داشته_باشد', 'خواهد_بود', 'طرف', 'تهیه', 'تبدیل', 'مناسب', 'زیرا', 'مشخص', 'می‌توانند', 'نزدیک', 'جریان', 'روند', 'بنابراین', 'می‌دهند', 'یافت', 'نخستین', 'بالا', 'پنج', 'ریزی', 'عالی', 'چیزی', 'نخست', 'بیشتری', 'ترتیب', 'شده_بود', 'خاص', 'خوبی', 'خوب', 'شروع', 'فرد', 'کامل', 'غیر', 'می‌رود', 'دهند', 'آخرین', 'دادن', 'جدی', 'بهترین', 'شامل', 'گیرد', 'بخشی', 'باشند', 'تمامی', 'بهتر', 'داده_است', 'حد', 'نبود', 'کسانی', 'می‌کرد', 'داریم', 'می‌باشد', 'دانست', 'ناشی', 'داشتند', 'دهه', 'می‌شد', 'ایشان', 'آنجا', 'گرفته_است', 'دچار', 'می‌آید', 'لحاظ', 'آنکه', 'داده', 'بعضی', 'هستیم', 'اند', 'برداری', 'نباید', 'می‌کنیم', 'نشست', 'سهم', 'همیشه', 'آمد', 'اش', 'وگو', 'می‌کنم', 'حداقل', 'طبق', 'جا', 'خواهد_کرد', 'نوعی', 'چگونه', 'رفت', 'هنگام', 'فوق', 'روش', 'ندارند', 'سعی', 'بندی', 'شمار', 'کلی', 'کافی', 'مواجه', 'همچنان', 'زیاد', 'سمت', 'کوچک', 'داشته_است', 'چیز', 'پشت', 'آورد', 'حالا', 'روبه', 'سال‌های', 'دادند', 'می‌کردند', 'عهده', 'نیمه', 'جایی', 'دیگران', 'سی', 'بروز', 'یکدیگر', 'آمده_است', 'جز', 'کنم', 'سپس', 'کنندگان', 'خودش', 'همواره', 'یافته', 'شان', 'صرف', 'نمی‌شود', 'رسیدن', 'چهارم', 'یابد', 'متر', 'ساز', 'داشته', 'کرده_بود', 'باره', 'نحوه', 'کردم', 'تو', 'شخصی', 'داشته_باشند', 'محسوب', 'پخش', 'کمی', 'متفاوت', 'سراسر', 'کاملا', 'داشتن', 'نظیر', 'آمده', 'گروهی', 'فردی', 'ع', 'همچون', 'خطر', 'خویش', 'کدام', 'دسته', 'سبب', 'عین', 'آوری', 'متاسفانه', 'بیرون', 'دار', 'ابتدا', 'شش', 'افرادی', 'می‌گویند', 'سالهای', 'درون', 'نیستند', 'یافته_است', 'پر', 'خاطرنشان', 'گاه', 'جمعی', 'اغلب', 'دوباره', 'می‌یابد', 'لذا', 'زاده', 'گردد', 'اینجا']


f=open("IR_data_news_12k.json")
data=json.load(f)

word_dictionary={}
inverted_index={}
positional_index={}

my_normalizer=Normalizer()
my_stemm=FindStems()

dictionary_size=1
tokens_size=1

heap_T5=[]
heap_M5=[]
heap_T=[]
heap_M=[]

# Preprocessing and creating indexes
for doc_id,doc in data.items():
    if doc_id=='500' or doc_id=='1000' or doc_id=='1500' or doc_id=='2000':
        heap_T5.append(tokens_size)
        heap_M5.append(dictionary_size)
    heap_T.append(tokens_size)
    heap_M.append(dictionary_size)
    print(f'indexing document : {doc_id}')
    word_index=0
    content_array=word_tokenize(my_normalizer.normalize(doc['content']))
    finalized_arr=[]
    for i in range(len(content_array)):
        #lemmatized=Lemmatizer().lemmatize(content_array[i])
        lemmatized=content_array[i]
        tmp=my_stemm.convert_to_stem(lemmatized)
        if tmp not in stop_words :
            finalized_arr.append(tmp)
    for word in finalized_arr:
        if word in word_dictionary:
            word_dictionary[word]['total_count']=word_dictionary[word]['total_count']+1
            if doc_id not in word_dictionary[word]:
                word_dictionary[word][doc_id]=0
            word_dictionary[word][doc_id]=word_dictionary[word][doc_id]+1
            if doc_id not in positional_index[word]:
                positional_index[word][doc_id]=[]    
            
            inverted_index[word].add(int(doc_id))
            bisect.insort(positional_index[word][doc_id],word_index)
        else:
            dictionary_size+=1
            word_dictionary[word]={}
            word_dictionary[word]['total_count']=1
            word_dictionary[word][doc_id]=1
            positional_index[word]={}
            positional_index[word][doc_id]=[]
            bisect.insort(positional_index[word][doc_id],word_index)
            inverted_index[word]=set()
            inverted_index[word].add(int(doc_id))
        tokens_size+=1
        word_index+=1



# Saving datas for loading faster for next uses
with open('positional.pkl', 'wb') as f:
    pickle.dump(positional_index, f)
with open('word_dictionary.pkl', 'wb') as f:
    pickle.dump(word_dictionary, f)
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
with open('inverted.pkl','wb') as f:
    pickle.dump(inverted_index,f)
    
print("Preprocessing finished")
    
# Checking Zipf Law
"""
sorted_by_total_count=[]
for key,value in word_dictionary.items():
    bisect.insort(sorted_by_total_count,value['total_count'])
sorted_by_total_count=sorted_by_total_count[::-1]
k=sorted_by_total_count[0]
X=[]
y_real=[]
y_pred=[]
for i in range(len(sorted_by_total_count)):
    cf_i_predicted=math.log10(k)-math.log10(i+1)
    cf_i_real=math.log10(sorted_by_total_count[i])
    X.append(math.log10(i+1))
    y_pred.append(cf_i_predicted)
    y_real.append(cf_i_real)
    
plt.plot(X,y_real,label='real')
plt.plot(X,y_pred,label='predicted')
plt.legend()
plt.show()
"""

# Checking Heaps' Law
"""
for i in range(len(heap_T5)):
    heap_T5[i]=math.log10(heap_T5[i])
    heap_M5[i]=math.log10(heap_M5[i])

for i in range(len(heap_T)):
    heap_T[i]=math.log10(heap_T[i])
    heap_M[i]=math.log10(heap_M[i])

heap_T=np.array(heap_T)    
heap_T5=np.array(heap_T5)
heap_M=np.array(heap_M)
heap_M5=np.array(heap_M5)


regression_model = LinearRegression()
regression_model.fit(heap_T5.reshape(-1,1),heap_M5)
m=regression_model.coef_
h=regression_model.predict(heap_T5.reshape(-1,1))[0]-m*heap_T5[0]

plt.plot(heap_T,m*heap_T+h,label='predicted')
plt.plot(heap_T,heap_M,label='real')
plt.legend()
plt.show()

print(10**(m*math.log10(tokens_size)+h))
print(f'k : {10**h} , b : {m}')
"""


def intersect(posting_a,posting_b):
    ans=[]
    index_a=0
    index_b=0
    while index_a!=len(posting_a) and index_b!=len(posting_b):
        if posting_a[index_a]==posting_b[index_b]:
            ans.append(posting_a[index_a])
            index_a+=1
            index_b+=1
        elif posting_a[index_a]<posting_b[index_b]:
            index_a+=1
        else:
            index_b+=1
    return ans




def phrase_operator(a,b):
    docs=intersect(list(sorted(inverted_index[a])),list(sorted(inverted_index[b])))
    ans=[]
    for doc in docs:
        posting_a=positional_index[a][str(doc)]
        posting_b=positional_index[b][str(doc)]
        index_a=0
        index_b=0
        while index_a!=len(posting_a) and index_b!=len(posting_b):
            if posting_a[index_a]==posting_b[index_b]-1:
                ans.append(doc)
                break
            elif posting_a[index_a]<posting_b[index_b]-1:
                index_a+=1
            else:
                index_b+=1
    return ans

def show_results(array):
    for i in array:
        title=data[str(i)]['title']
        url=data[str(i)]['url']
        print(f"موضوع : {title}")
        print(f"آدرس : {url}")
        print("*******************************************************")
    if len(array)==0:
        print("No result")
        
def sort_results(results,words):
    tmp_results=[]
    for doc in results:
        score=0
        for word in words:
            score+=word_dictionary[word][str(doc)]
        tmp_results.append((doc,score))
    tmp_results=sorted(tmp_results, key = lambda x: x[1])
    final_results=[]
    for i in tmp_results:
        final_results.append(i[0])
    final_results=final_results[::-1]
    return final_results
    

# Creating Champions list
champion_index={}
r=50
for i in inverted_index:
    new_set=set()
    tmp=[]
    docs=inverted_index[i]
    for doc_id in docs:
        weight=1+math.log(word_dictionary[i][str(doc_id)])
        tmp.append((doc_id,weight))
    tmp=sorted(tmp, key = lambda x: x[1])

    for j in range(min(len(tmp),r)):
        new_set.add(tmp[len(tmp)-j-1][0])
    champion_index[i]=new_set


def cosine_score(query_array):
    """
    Cosine similarity function
    """
    scores={}
    lengths={}
    for term in query_array:
        posting_list=inverted_index[term]
        if len(posting_list)>11000:
            continue
        w_t_q=1*math.log(12165/len(posting_list))
        for doc_id in champion_index[term]:
            if doc_id not in scores:
                scores[doc_id]=0
            if doc_id not in lengths:
                lengths[doc_id]=0
            w_t_d=(1+math.log(word_dictionary[term][str(doc_id)]))*(math.log(12165/len(posting_list)))
            scores[doc_id]+=w_t_q*w_t_d
            lengths[doc_id]+=w_t_d*w_t_d
    final_scores=[]
    for doc_id,score in scores.items():
        final_scores.append((doc_id,(scores[doc_id]/(math.sqrt(lengths[doc_id])))))
    final_results=sorted(final_scores, key = lambda x: x[1])
    ans=[]
    k=4
    for i in range(min(k,len(final_results))):
        ans.append(final_results[len(final_results)-i-1][0])
    show_results(ans)





def process_input(query):
    """
    getting query and search it in indexes and showing the results
    """
    query=query
    phrase=""
    and_words=[]
    not_word=""
    and_result=[]
    not_result=[]
    phrase_result=[]
    final_res=[]
    if '"' in query:
        phrase=query.split('"')[1]
        query=query.replace(phrase,"")
        query=query.replace('"',"")
    query=word_tokenize(Normalizer().normalize(query))
    for i in range(len(query)):
        query[i]=FindStems().convert_to_stem(query[i])
    cosine_score(query)

        
        
print("Enter a query:")
user_input=input()
while user_input!="exit":
    try:
        process_input(user_input)
    except:
        print("no result")
    print("---------------------------------------------------------------------")
    print("Enter new query:")
    user_input=input()
