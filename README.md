
# Hebrew-Sentence-Simplification

Simplification of Hebrew texts for Israel state institutions using Random Forest Classification with Scikit-Learn.

Final project in a software engineering degree.

![App Banner](https://raw.githubusercontent.com/Shira-Malka/Hebrew-Sentence-Simplification/master/app-banner.png)


## API Reference

#### YAP Parser using LangNData

```https://www.langndata.com/heb_parser/register```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_token` | `string` | **Required**. Your API token |



## Documentation

[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[YAP Hebrew Parser](https://www.langndata.com/heb_parser/api_reference)


## Screenshots

Input: ״באתר זה תוכל לברר את הזכויות המגיעות לך בהתאם לאירועים ושינויים שהתרחשו בחייך.״
![App Screenshot](https://raw.githubusercontent.com/Shira-Malka/Hebrew-Sentence-Simplification/master/img/complex.png)
  Output: detected as Complex.

Input: ״אתה מוזמן לפנות אלינו כדי למצות את זכויותיך.״
![App Screenshot](https://raw.githubusercontent.com/Shira-Malka/Hebrew-Sentence-Simplification/master/img/notComplex.png)
  Output: detected as Not complex.

Now enter any sentence in Hebrew and check it yourself!


## 🚀 About Us
Shira Malka - Backend Engineer

Ron Bar Zvi - Software Engineer


## Support

For support, email malka.shiraa@gmail.com | ron.bzeve@gmail.com



