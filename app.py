import itertools
import re
import time

import requests
import pandas as pd

from collections import Counter
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def sentenceSimplification():
    """
    This function build the sentence simplification vector with parts of speech.
    :return:
    """
    s = sentenceGrid.get()

    words = []
    letters = []
    word_avg = []
    numbers = []
    symbols = []
    if_question = []

    punctuation = []
    sym_hashtag = []
    sym_shtrudel = []
    sym_dollar = []
    sym_ampersand = []
    sym_percend = []
    sym_star = []
    sym_tilda = []
    sym_math = []
    sym_logic = []
    sym_brackets = []
    sym_num = []
    sym_quotes = []
    sym_slashes = []

    bigrams = []
    t_depth = []
    mdd = []
    num_of_stopwords = []
    result = []

    spl = s.split()
    bigrams.append(len(spl) - 1)

    val = re.findall(r'\w+', s)
    words.append(len(val))

    val = re.findall(r'\w', s)
    letters.append(len(val))

    val = re.findall(r'\w+', s)
    average = sum(len(word) for word in val) / len(val)
    word_avg.append(round(average, 3))

    val = re.findall(r'[0-9]', s)
    numbers.append(len(val))

    val = re.findall(r'[^א-ת0-9 ]', s)
    symbols.append(len(val))

    val = re.findall(r'[`!^:;\',.?]', s)
    punctuation.append(len(val))

    val = re.findall(r'[#]', s)
    sym_hashtag.append(len(val))

    val = re.findall(r'[@]', s)
    sym_shtrudel.append(len(val))

    val = re.findall(r'[$]', s)
    sym_dollar.append(len(val))

    val = re.findall(r'[&]', s)
    sym_ampersand.append(len(val))

    val = re.findall(r'[%]', s)
    sym_percend.append(len(val))

    val = re.findall(r'[*]', s)
    sym_star.append(len(val))

    val = re.findall(r'[~]', s)
    sym_tilda.append(len(val))

    val = re.findall(r'[\+\-\*\/<>=%^]', s)
    sym_math.append(len(val))

    val = re.findall(r'[&|]', s)
    sym_logic.append(len(val))

    val = re.findall(r'[()[]{}]', s)
    sym_brackets.append(len(val))

    val = re.findall(r'[№]', s)
    sym_num.append(len(val))

    val = re.findall(r'["\']', s)
    sym_quotes.append(len(val))

    val = re.findall(r'[\//]', s)
    sym_slashes.append(len(val))

    if s[len(s) - 1] == '?':
        if_question.append(1)
    else:
        if_question.append(0)

    NN_l = []
    VB_l = []
    JJ_l = []
    PREPOSITION_l = []
    DEF_l = []
    CONJ_l = []
    BN_l = []
    NNP_l = []
    NNT_l = []
    TEMP_l = []
    RB_l = []
    CC_l = []
    AT_l = []
    QW_l = []
    REL_l = []
    PRP_l = []
    IN_l = []
    DTT_l = []
    DT_l = []
    MD_l = []

    s = s.replace('[^\w\s]', '')
    s = s.replace('[^א-ת ]', '')

    text = re.sub("\"", "", s)

    url = <PUT_YOUR_TOKEN_HERE>
    data = '{"data":"' + text + '"}'
    headers = {'content-type': 'application/json'}
    response = requests.post(url, data=data.encode('utf-8'), headers=headers)
    json_response = response.json()

    heads_tree = []
    heads = []
    dep_tree = json_response['dep_tree']

    for dict_ in dep_tree.values():
        inner_tree = dict_
        for i in range(len(inner_tree)):
            if list(inner_tree)[7] == '_' and list(inner_tree)[8] == '_':
                heads_tree.append(dict(itertools.islice(inner_tree.items(), 9)))
                inner_tree = dict(itertools.islice(inner_tree.items(), 9, None))
            elif list(inner_tree)[8] == '_' and list(inner_tree)[9] == '_':
                heads_tree.append(dict(itertools.islice(inner_tree.items(), 10)))
                inner_tree = dict(itertools.islice(inner_tree.items(), 10, None))
            else:
                heads_tree.append(dict(itertools.islice(inner_tree.items(), 9)))

    for i in range(len(heads_tree)):
        if heads_tree[i]['dependency_arc'].isnumeric():
            heads.append(int(heads_tree[i]['dependency_arc']))
        elif heads_tree[i]['empty'].isnumeric():
            heads.append(int(heads_tree[i]['empty']))

    t_depth.append(max(heads))
    if (len(heads) - 1) == 0:
        mdd.append(float("{:.2f}".format(sum(heads))))
    else:
        mdd.append(float("{:.2f}".format(sum(heads) / (len(heads) - 1))))

    NN = 0
    VB = 0
    JJ = 0
    PREPOSITION = 0
    DEF = 0
    CONJ = 0
    BN = 0
    NNP = 0
    NNT = 0
    TEMP = 0
    RB = 0
    CC = 0
    AT = 0
    QW = 0
    REL = 0
    PRP = 0
    IN = 0
    DTT = 0
    DT = 0
    MD = 0

    for dict_ in dep_tree.values():
        NN = NN + Counter(dict_.values())['NN']
        VB = VB + Counter(dict_.values())['VB']
        JJ = JJ + Counter(dict_.values())['JJ']
        PREPOSITION = PREPOSITION + Counter(dict_.values())['PREPOSITION']
        DEF = DEF + Counter(dict_.values())['DEF']
        CONJ = CONJ + Counter(dict_.values())['CONJ']
        BN = BN + Counter(dict_.values())['BN']
        NNP = NNP + Counter(dict_.values())['NNP']
        NNT = NNT + Counter(dict_.values())['NNT']
        TEMP = TEMP + Counter(dict_.values())['TEMP']
        RB = RB + Counter(dict_.values())['RB']
        CC = CC + Counter(dict_.values())['CC']
        AT = AT + Counter(dict_.values())['AT']
        QW = QW + Counter(dict_.values())['QW']
        REL = REL + Counter(dict_.values())['REL']
        PRP = PRP + Counter(dict_.values())['PRP']
        IN = IN + Counter(dict_.values())['IN']
        DTT = DTT + Counter(dict_.values())['DTT']
        DT = DT + Counter(dict_.values())['DT']
        MD = MD + Counter(dict_.values())['MD']

    NN_l.append(NN)
    VB_l.append(VB)
    JJ_l.append(JJ)
    PREPOSITION_l.append(PREPOSITION)
    DEF_l.append(DEF)
    CONJ_l.append(CONJ)
    BN_l.append(BN)
    NNP_l.append(NNP)
    NNT_l.append(NNT)
    TEMP_l.append(TEMP)
    RB_l.append(RB)
    CC_l.append(CC)
    AT_l.append(AT)
    QW_l.append(QW)
    REL_l.append(REL)
    PRP_l.append(PRP)
    IN_l.append(IN)
    DTT_l.append(DTT)
    DT_l.append(DT)
    MD_l.append(MD)

    stopwords = ['אני', 'את', 'אתה', 'אנחנו', 'אתן', 'אתם', 'הם', 'הן', 'היא', 'הוא', 'שלי', 'שלו', 'שלך', 'שלה', 'שלנו', 'שלכם', 'שלכן', 'שלהם', 'שלהן', 'לי',
                 'לו', 'לה', 'לנו', 'לכם', 'לכן', 'להם', 'להן', 'אותה', 'אותו', 'זה', 'זאת', 'אלה', 'אלו', 'תחת', 'מתחת', 'מעל', 'בין', 'עם', 'עד', 'נגר', 'על', 'אל', 'מול', 'של',
                 'אצל', 'כמו', 'אחר', 'אותו', 'בלי', 'לפני', 'אחרי', 'מאחורי', 'עלי', 'עליו', 'עליה', 'עליך', 'עלינו', 'עליכם', 'לעיכן', 'עליהם', 'עליהן', 'כל', 'כולם', 'כולן',
                 'כך',
                 'ככה', 'כזה', 'זה', 'זות', 'אותי', 'אותה', 'אותם', 'אותך', 'אותו', 'אותן', 'אותנו', 'ואת', 'את', 'אתכם', 'אתכן', 'איתי', 'איתו', 'איתך', 'איתה', 'איתם', 'איתן',
                 'איתנו', 'איתכם', 'איתכן', 'יהיה', 'תהיה', 'היתי', 'היתה', 'היה', 'להיות', 'עצמי', 'עצמו', 'עצמה', 'עצמם', 'עצמן', 'עצמנו', 'עצמהם', 'עצמהן', 'מי', 'מה', 'איפה',
                 'היכן', 'אם', 'לאן', 'איזה', 'מהיכן', 'איך', 'כיצד', 'מתי', 'כאשר', 'כש', 'למרות',
                 'לפני', 'אחרי', 'למה', 'מדוע', 'כי', 'יש', 'אין', 'אך', 'מנין', 'מאין', 'מאיפה', 'יכל', 'יכלה', 'יכלו', 'יכול', 'יכולה', 'יכולים', 'יכולות', 'יוכלו', 'יוכל',
                 'מסוגל',
                 'לא', 'רק', 'אולי', 'אין', 'לאו', 'אי', 'כלל', 'נגד', 'אם', 'עם', 'אל', 'אלה', 'אלו', 'אף', 'על', 'מעל', 'מתחת', 'מצד', 'בשביל', 'לבין', 'באמצע', 'בתוך', 'דרך',
                 'מבעד', 'באמצעות', 'למעלה', 'למטה', 'מחוץ', 'מן', 'לעבר', 'מכאן', 'כאן', 'הנה', 'הרי', 'פה', 'שם', 'אך', 'ברם', 'שוב', 'אבל', 'מבלי', 'בלי', 'מלבד', 'רק',
                 'בגלל', 'מכיוון', 'עד', 'אשר', 'ואילו', 'למרות', 'כמו', 'כפי', 'אז', 'אחרי', 'כן', 'לכן', 'לפיכך', 'מאד', 'מעט', 'מעטים', 'במידה', 'שוב', 'יותר', 'מדי', 'גם',
                 'כן',
                 'נו', 'אחר', 'אחרת', 'אחרים', 'אחרות', 'אשר', 'או']

    splitSentence = s.split()
    stopWordsNum = 0
    for word in splitSentence:
        if word in stopwords:
            stopWordsNum = stopWordsNum + 1
    num_of_stopwords.append(stopWordsNum)

    sentenceVector = pd.DataFrame({
        'mdd': mdd,
        't_depth': t_depth,
        'bigrams': bigrams,
        'words': words,
        'letters': letters,
        'word_avg': word_avg,
        'numbers': numbers,
        'symbols': symbols,
        'if_question': if_question,
        'punctuation': punctuation,
        'sym_hashtag': sym_hashtag,
        'sym_shtrudel': sym_shtrudel,
        'sym_dollar': sym_dollar,
        'sym_ampersand': sym_ampersand,
        'sym_percend': sym_percend,
        'sym_star': sym_star,
        'sym_tilda': sym_tilda,
        'sym_math': sym_math,
        'sym_logic': sym_logic,
        'sym_brackets': sym_brackets,
        'sym_num': sym_num,
        'sym_quotes': sym_quotes,
        'sym_slashes': sym_slashes,
        'NN': NN_l,
        'VB': VB_l,
        'JJ': JJ_l,
        'PREPOSITION': PREPOSITION_l,
        'DEF': DEF_l,
        'CONJ': CONJ_l,
        'BN': BN_l,
        'NNP': NNP_l,
        'NNT': NNT_l,
        'TEMP': TEMP_l,
        'RB': RB_l,
        'CC': CC_l,
        'AT': AT_l,
        'QW': QW_l,
        'REL': REL_l,
        'PRP': PRP_l,
        'IN': IN_l,
        'DTT': DTT_l,
        'DT': DT_l,
        'MD': MD_l,
        'stopwords': num_of_stopwords})

    #############

    data = pd.read_csv('./data/HebrewSentences.csv')
    X = data.drop(['sentence', 'result'], axis=1)
    y = data['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)

    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)
    y_pred = rf.predict(sentenceVector)

    if y_pred == 1:
        complex = Label(userUI, text='Your sentence:\n\n"' + sentenceGrid.get() + '"\n\n detected as: COMPLEX.')
        complex.grid(row=4, column=1)
    else:
        notComplex = Label(userUI, text='Your sentence:\n\n"' + sentenceGrid.get() + '"\n\n detected as: NOT COMPLEX.')
        notComplex.grid(row=4, column=1)


userUI = Tk()
userUI.title('Hebrew Sentence Detection')
userUI.geometry("900x500")


kav = Label(userUI, text='                ')
kav.grid(row=0, column=0)

headerLabel = Label(userUI, text='\nHEBREW COMPLEX SENTENCE DETECTION\n', font=("Arial", 25))

headerLabel.grid(row=0, column=1)
kav = Label(userUI, text='------------------------------' +
                         '--------------------------------------------------------------')
kav.grid(row=1, column=1)

kav = Label(userUI, text='      ')
kav.grid(row=1, column=2)

runModelButton = Button(userUI, text='RUN \nMODEL', width=15, height=5, pady=10, padx=10, fg='blue', bg='yellow', command=sentenceSimplification)
runModelButton.grid(row=2, column=3)

kav2 = Label(userUI, text='                ')
kav2.grid(row=3, column=0)

exitButton = Button(userUI, text='EXIT', width=15, height=5, pady=10, padx=10, fg='blue', bg='yellow', command=userUI.destroy)
exitButton.grid(row=4, column=3)

sentenceGrid = Entry(userUI, width=50, bg='lightblue', borderwidth=5)
sentenceGrid.insert(0, 'Write sentence to check...')
sentenceGrid.grid(row=2, column=1)

userUI.mainloop()
