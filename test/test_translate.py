import unittest

from indictrans2_ct2_inference.translate import Translator


class TestBatchTranslate(unittest.TestCase):
    SRC_LANG = "ur"
    TRG_LANG = "en"

    INPUT_SENTENCES = [
"انہوں نے مزید بتایا کہ، \"اب ہمارے پاس غیر ذیابیس والے 4 مہینے کی عمر کے چوہے ہیں جنہیں شوگر ہوجایا کرتا تھا۔\"",
"ہالیفیکس، نووا اسکاٹیا کی ڈلہوزی یونیورسٹی میں طب کے پروفیسر اور کینیڈین ڈائبیٹک ایسوسی ایشن کے طبی و سائنسی ڈویژن کے صدر ڈاکٹر ایہود یوآر نے متنبہ کیا ہے کہ تحقیق ابھی تک اپنے ابتدائی مراحل میں ہے۔",
"کچھ دوسرے ماہرین کی طرح سے، انہیں اس بارے میں شک ہے کہ آیا ذیابیطس کا علاج کیا جا سکتا ہے، اس بات کو نوٹ کرتے ہوئے کہ ان نتائج ۔کی ایسے لوگوں سے کوئی مطابقت نہیں ہے جو پہلے سے ٹائپ 1 ذیابیطس میں مبتلا ہیں۔",
"نوبل کمیٹی برائے ادب کی مستقل سکریٹری سارا ڈینیئس نے پیر کو سویڈش اکیڈمی میں سیرجیس ریڈیو، سوڈان پر منعقد ایک ریڈیو پروگرام کے دوران عوامی سطح پر یہ اعلان کیا کہ نوبل انعام برائے ادب 2016 جیتنے والے باب ڈائلن تک براہ راست رسائی حاصل نہ کر پانے کی وجہ سے کمیٹی نے ان تک پہنچنے کی کوشش ترک کر دی۔",
"ڈینئس نے کہا، “فی الحال ہم کچھ نہیں کر رہے۔ میں نے اس کے قریب ترین شریک کار کو کال کی ہے اور ای میلز بھیجی ہیں اور بہت دوستانہ جوابات موصول کیے ہیں۔ فی الوقت، یقیناً اتنا ہی کافی ہے۔”",
"اس سے قبل رِنگ کے سی ای او جیمی سمیناف نے تبصرہ کیا تھا کہ کمپنی اس وقت شروع کی گئی تھی جب ان کے دروازے کی گھنٹی ان کی دکان سے گیرج میں نہیں سنائی دیتی تھی۔",
"اس نے کہا کہ اس نے ایک وائی فائی ڈور بیل اختراع کیا ہے۔",
"سیمینوف نے کہا کہ اس کے 2013 کی شارک ٹینک والی قسط میں ظاہر ہونے کے بعد فروخت میں اضافہ ہوا جہاں شو کے پینل نے آغاز کے لیے فنڈ دینے سے انکار کر دیا تھا۔",
"2017 کے آخر میں سمینوف خرید و فروخت کے ٹی وی چینل کيو وی سی پر حاضر ہوئے۔",
"Ring نے مسابق سیکیورٹی کمپنی ADT Corporation کے ساتھ اپنے قانونی دعوے کا تسویہ بھی کر لیا۔",
    ]

    EXPECTED_TRANSLATIONS = [
"\"We now have non-diabetic 4-month-old mice that were fed sugar,\" he added.",
"Dr. Ehud UR, professor of medicine at Dalhousie University in Halifax, Nova Scotia, and president of the medical and scientific division of the Canadian Diabetic Association, warns that research is still in its early stages.",
"Like some other experts, he is skeptical about whether diabetes can be cured, noting that the findings have no relevance to people who already have type 1 diabetes.",
"Sara Danius, the permanent secretary of the Nobel Committee for Literature, publicly announced during a radio program hosted on Sirijis Radio, Sudan, at the Swedish Academy on Monday that the committee abandoned its attempt to reach Bob Dylan, the 2016 Nobel Prize in Literature winner, because it could not directly reach him.",
"\"At the moment we are not doing anything; I have called and sent emails to his closest associate and received very friendly responses.",
"Ring CEO Jamie Semenoff previously commented that the company was started when their doorbell wouldn't ring in the garage from their store.",
"He said he had invented a Wi-Fi doorbell.",
"Semenov stated that sales increased after he appeared in a 2013 Shark Tank episode where the show's panel refused to fund the launch.",
"In late 2017, Semenov appeared on the TV shopping channel QVC.",
"Ring also settled its legal claim with competing security company ADT Corporation.",
    ]

    @classmethod
    def setUpClass(cls):
        cls.translator = Translator(
            src_lang=cls.SRC_LANG,
            trg_lang=cls.TRG_LANG,
            device_index=[0],
            mini_batch_size=1000,   # batch fits in a single mini-batch; change if you like
            beam_size=4,
        )

    def test_batch_translate_matches_expected(self):
        self.assertEqual(
            len(self.INPUT_SENTENCES), 10,
            "INPUT_SENTENCES must contain exactly 10 sentences."
        )
        self.assertEqual(
            len(self.EXPECTED_TRANSLATIONS), 10,
            "EXPECTED_TRANSLATIONS must contain exactly 10 translations."
        )

        outputs = self.translator.batch_translate(self.INPUT_SENTENCES)

        self.assertEqual(
            len(outputs), 10,
            "batch_translate should return exactly 10 translations."
        )

        for i, (actual, expected) in enumerate(zip(outputs, self.EXPECTED_TRANSLATIONS), start=1):
            with self.subTest(sentence_index=i):
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
