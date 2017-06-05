import textblob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import NLTKClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

import os


class textCat(object):
    def __init__(self):
        self.no = 0
        self.train_path = './0001TrainText/'
        self.test_path = './0002TestText/'
        self.label_path = './0003Labels/'

        self.files_0001 = os.listdir(self.train_path)
        self.files_0002 = os.listdir(self.test_path)
        self.files_0003 = os.listdir(self.label_path)

        self.train_dic = {}
        self.test_dic = {}
        self.train_corpus = []
        self.test_corpus = []
        self.train_target = []
        self.test_target = []
        self.tv_train = None
        self.tv_test = None

        for local_i in range(len(self.files_0001)):
            with open(self.train_path + self.files_0001[local_i]) as local_of:
                local_content = local_of.read()
                # self.train_corpus.append(local_content)
                # print files_0001[i], type(files_0001[i])
                self.train_dic[int(self.files_0001[local_i])] = local_content

        for local_i in range(len(self.files_0002)):
            with open(self.test_path + self.files_0002[local_i]) as local_of:
                local_content = local_of.read()
                # self.test_corpus.append(local_content)
                # print files_0001[i], type(files_0001[i])
                self.test_dic[int(self.files_0002[local_i])] = local_content

        # print len(train_dic)
        self.train_label = []
        self.test_label = []
        with open(self.label_path + "test.doc.label") as local_of:
            print self.files_0003[0]
            local_content = local_of.readlines()
            for local_i in range(len(local_content)):
                local_ctt_splt = local_content[local_i].split('\t')
                # print test_dic[int(ctt_splt[0])]
                # self.test_target.append(int(local_ctt_splt[1]))
                self.test_label.append((int(local_ctt_splt[0]), int(local_ctt_splt[1])))

        with open(self.label_path + "train.doc.label") as local_of:
            print self.files_0003[1]
            local_content = local_of.readlines()
            for local_i in range(len(local_content)):
                local_ctt_splt = local_content[local_i].split('\t')
                # print test_dic[int(ctt_splt[0])]
                # self.train_target.append(int(local_ctt_splt[1]))
                # train_label is (filename, target label)
                self.train_label.append((int(local_ctt_splt[0]), int(local_ctt_splt[1])))
        pass

    def get_ready(self):
        for i in range(len(self.train_label)):
            self.train_corpus.append(self.train_dic[self.train_label[i][0]])
            self.train_target.append(self.train_label[i][1])
        for i in range(len(self.test_label)):
            self.test_corpus.append(self.test_dic[self.test_label[i][0]])
            self.test_target.append(self.test_label[i][1])
        pass

    def fea_extract(self):
        tv_model_1 = TfidfVectorizer(sublinear_tf=True, max_df=0.7, min_df=9, stop_words='english')
        self.tv_train = tv_model_1.fit_transform(self.train_corpus)
        tv_model_2 = TfidfVectorizer(vocabulary=tv_model_1.vocabulary_)
        self.tv_test = tv_model_2.fit_transform(self.test_corpus)
        pass

    def svm_cls(self, reverse=False):
        if not reverse:
            svc_cf = SVC(kernel='linear')
            svc_cf.fit(self.tv_train, ins_tcat.train_target)
            pred = svc_cf.predict(self.tv_test)
            calculate_result(self.test_target, pred)
        else:
            svc_cf = SVC(kernel='linear')
            svc_cf.fit(self.tv_test, ins_tcat.test_target)
            pred = svc_cf.predict(self.tv_train)
            calculate_result(self.train_target, pred)
        pass

    def calc_prec(self):
        pass

    def test_1(self):
        l_vectorizer = CountVectorizer(min_df=1)

        corpus = [
            'This is the first\n document.',
            'This is the second\n                    second document.',
            'And the third one.',
            'Is this the first document?',
        ]
        l_X = l_vectorizer.fit_transform(corpus)
        feature_name = l_vectorizer.get_feature_names()

        print feature_name
        print l_X.toarray()

    def main(self):
        pass


def calculate_result(actual, f_pred):
    if len(actual) != len(f_pred):
        print 'there is some wrong!'
    err = 0
    for i in range(len(actual)):
        if actual[i] != f_pred[i]:
            err += 1
    m_pre = 1. * (len(actual) - err) / len(actual)
    print 'predict info:'
    print 'precision: ', round(m_pre, 3)

    pass


if __name__ == '__main__':
    ins_tcat = textCat()
    ins_tcat.get_ready()
    ins_tcat.fea_extract()
    ins_tcat.svm_cls(reverse=True)
    # ins_tcat.test_1()
    """
    Here could be more fix, for |feature| == 2
    """

    # tv = TfidfVectorizer(sublinear_tf=True, max_df=0.7, min_df=9, stop_words='english')
    # tv_train = tv.fit_transform(ins_tcat.train_corpus)
    # tv2 = TfidfVectorizer(vocabulary=tv.vocabulary_)
    # tv_test = tv2.fit_transform(ins_tcat.test_corpus)
    # print repr(tv_train.shape), repr(tv_test.shape)

    # svc_cf = SVC(kernel='linear')
    # svc_cf.fit(tv_train, ins_tcat.train_target)
    # pred = svc_cf.predict(tv_test)
    # calculate_result(ins_tcat.test_target, pred)

