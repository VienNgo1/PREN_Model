from Nets.model import Model
from Utils.utils import *
from Configs.testConf import configs
from data.dataset import TestLoader
from Performance_error_analysis import plot_csv_file, view_images
import numpy as np

# Performance and analysis
import csv
import shutil
import pickle
import datetime
import itertools
import random as rn
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import warnings
import matplotlib.image as mpimg
warnings.filterwarnings("ignore")
# write the pred values to a file named 'predictions.txt'
output_file = 'predictions.txt'
# Define the file path for the CSV file
csv_file = 'output.csv'


# Performance function

from progressbar import *


class Tester(object):

    def __init__(self, model, testloader):

        self.device = torch.device('cuda' if configs.cuda else 'cpu')

        self.model = model.to(self.device)
        self.model.eval()

        self.testloader = testloader

        with open(configs.alphabet) as f:
            alphabet = f.readline().strip()
        self.converter = strLabelConverter(alphabet)


    def vert_val(self):
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=10 * len(self.testloader)).start()

        n_correct = 0.
        n_ims = 0

        with torch.no_grad():
            with open(output_file, 'w') as f:
                for step, (ims, texts, ims_clock, ims_counter, is_vert, imgpath) in enumerate(self.testloader):
                    img_path = imgpath[0]
                    ims = ims.to(self.device)
                    # print(ims)
                    logits = self.model(ims)  # [1, L, n_class]

                    if is_vert[0]:
                        ims_clock = ims_clock.to(self.device)
                        ims_counter = ims_counter.to(self.device)
                        logits_clock = self.model(ims_clock)
                        logits_counter = self.model(ims_counter)

                        score, pred = logits[0].log_softmax(1).max(1)  # [L]
                        pred = list(pred.cpu().numpy())
                        score_clock, pred_clock = logits_clock[0].log_softmax(1).max(1)
                        pred_clock = list(pred_clock.cpu().numpy())
                        score_counter, pred_counter = logits_counter[0].log_softmax(1).max(1)
                        pred_counter = list(pred_counter.cpu().numpy())

                        scores = np.ones(3) * -np.inf

                        if 1 in pred:
                            score = score[:pred.index(1)]
                            scores[0] = score.mean()
                        if 1 in pred_clock:
                            score_clock = score_clock[:pred_clock.index(1)]
                            scores[1] = score_clock.mean()
                        if 1 in pred_counter:
                            score_counter = score_counter[:pred_counter.index(1)]
                            scores[2] = score_counter.mean()

                        c = scores.argmax()
                        if c == 0:
                            pred = pred[:pred.index(1)]
                        elif c == 1:
                            pred = pred_clock[:pred_clock.index(1)]
                        else:
                            pred = pred_counter[:pred_counter.index(1)]

                    else:
                        pred = logits[0].argmax(1)
                        pred = list(pred.cpu().numpy())
                        if 1 in pred:
                            pred = pred[:pred.index(1)]

                    pred = self.converter.decode(pred)
                    pred = pred.replace('<unk>', '')
                    gt = texts[0]
                    n_correct += (pred == gt)
                    n_ims += 1

                    if configs.display:
                        print('{} ==> {}  {}'
                              .format(gt, pred, '' if pred == gt else 'error'))
                    # Write the pred value to the file
                    f.write('({})({})({})\n'.format(img_path, gt, pred))
                    progress.update(10 * step + 1)
                progress.finish()

            print('-' * 50)
            print('Acc_word = {:.3f}%'.format(100 * n_correct / n_ims))


    def val(self):
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        progress = ProgressBar(widgets=widgets, maxval=10 * len(self.testloader)).start()

        n_correct = 0.
        n_ims = 0

        self.model.eval()
        with torch.no_grad():

            for step, (ims, texts, *_) in enumerate(self.testloader):

                ims = ims.to(self.device)
                logits = self.model(ims)  # [B, L, n_class]
                preds = logits.argmax(2)  # [B, L]

                for pred, gt in zip(preds, texts):
                    pred = list(pred.cpu().numpy())
                    if 1 in pred:
                        pred = pred[:pred.index(1)]
                    pred = self.converter.decode(pred)
                    pred = pred.replace('<unk>', '')
                    n_correct += (pred == gt)
                    n_ims += 1

                    if configs.display:
                        print('{} ==> {}  {}'
                             .format(gt, pred, '' if pred == gt else 'error'))

                progress.update(10 * step + 1)
            progress.finish()

        print('-'*50)
        print('Acc_word = {:.3f}%'.format(100 * n_correct / n_ims))


def performance_analysis():
    # Initialize lists to store values
    img_paths = []
    ground_truths = []
    predicts = []

    # Open the file for reading
    with open(output_file, 'r') as file:
        # Read each line from the file
        for line in file:
            # Extract values between parentheses using string slicing
            img_path = line[line.find('(') + 1:line.find(')')]
            gt = line[line.find(')(') + 2:line.rfind(')(')]
            pred = line[line.rfind('(') + 1:line.rfind(')')]

            # Append extracted values to the respective lists
            img_paths.append(img_path)
            ground_truths.append(gt)
            predicts.append(pred)

    # Combine ground_truths and predicts into rows
    rows = zip(img_paths, ground_truths, predicts)

    # Write the rows to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_paths', 'ground_truths', 'predicts'])  # Write header
        writer.writerows(rows)

    # Read the CSV file into a DataFrame with the specified encoding
    test_df = pd.read_csv(csv_file, encoding='latin-1')

    # Mismatch count
    # mismatch_count = []
    #
    # for idx, row in enumerate(test_df.values):
    #
    #     original = str(row[1])
    #     predicted = str(row[2])
    #     num_mismatch = 0
    #
    #     for i in range(min(len(original), len(predicted))):
    #
    #         if original[i] != predicted[i]:
    #             num_mismatch += 1
    #
    #     num_mismatch += abs(
    #         len(predicted) - len(original))  # Considering the extra length of the predicted label also as mismatch
    #     mismatch_count.append(num_mismatch)
    # test_df['mismatch_count'] = mismatch_count
    # # print(test_df.head())
    # plt.figure(figsize=(15, 6))
    # ax = sns.countplot(x='mismatch_count', data=test_df, palette='Dark2')
    #
    # plt.title("\nDistribution of Character Mismatch Count\n")
    # plt.xlabel("Character Mismatch Count")
    # plt.ylabel("Word Count")
    #
    # plt.margins(0.05, 0.1)
    #
    # for p in ax.patches:
    #     x = p.get_bbox().get_points()[:, 0]
    #     y = p.get_bbox().get_points()[1, 1]
    #     ax.annotate('{:.1f}%'.format(100. * y / len(test_df)), (x.mean(), y + 1000),
    #                 ha='center', va='bottom')
    #
    # plt.show()

    # Mismatch percentage / Error Analysis
    mismatch_percentage = []

    for idx, row in enumerate(test_df.values):
        original = str(row[1])
        predicted = str(row[2])
        num_mismatch = 0

        for i in range(min(len(original), len(predicted))):
            if original[i] != predicted[i]:
                num_mismatch += 1

        num_mismatch += abs(len(original) - len(predicted))

        percentage = int(np.round((num_mismatch / len(original)) * 100))

        mismatch_percentage.append(percentage)
    test_df['mismatch_percentage'] = mismatch_percentage
    # plot_csv_file(test_df, 'mismatch_percentage', "\nDistribution of the Character Mismatch Percentage\n", "Character Mismatch Percentage", "Word Count")
    # Character Mismatch Percentage <= 15% --- BEST CASE
    # 15% < Character Mismatch Percentage <= 40% --- AVERAGE CASE
    # 40% < Character Mismatch Percentage --- WORST CASE

    test_df['case'] = [""] * len(test_df)  # Adding the case column
    test_df['case'] = ["best" if p <= 15 else test_df['case'][idx] for idx, p in enumerate(test_df['mismatch_percentage'].values)]
    test_df['case'] = ["average" if (p > 15) and (p <= 40) else test_df['case'][idx] for idx, p in enumerate(test_df['mismatch_percentage'].values)]
    test_df['case'] = ["worst" if p > 40 else test_df['case'][idx] for idx, p in enumerate(test_df['mismatch_percentage'].values)]

    best_case_images = test_df[test_df['case']=='best']['image_paths'].index.values
    average_case_images = test_df[test_df['case'] == 'average']['image_paths'].index.values
    worst_case_images = test_df[test_df['case'] == 'worst']['image_paths'].index.values

    # rn.shuffle(best_case_images)
    # view_images(csv_file, best_case_images, 'Best Case', 10)
    # rn.shuffle(average_case_images)
    # view_images(csv_file, average_case_images, 'Average Case', 10)
    rn.shuffle(worst_case_images)
    view_images(csv_file, worst_case_images, 'Worst Case', 15)


# def plot_csv_file(test_df, head,title, xlabel, ylabel):
#     plt.figure(figsize=(16, 20))
#     ax = sns.barplot(x=test_df[head].value_counts().index,
#                      y=test_df[head].value_counts().values, ci=False, palette='Accent',
#                      orient='h')
#
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     plt.margins(0.1, 0.01)
#
#     for p in ax.patches:
#         width = p.get_width()
#         plt.text(4 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
#                  '{:1.1f}%'.format(width),
#                  ha='center', va='center')
#     plt.show()


# def view_images(indices, case, size):
#     test_df = pd.read_csv(csv_file, encoding='latin-1')
#     paths = test_df.image_paths.values[indices]
#     labels = test_df.ground_truths.values[indices]
#     predictions = test_df.predicts.values[indices]
#
#     fig = plt.figure(figsize=(15, 15))
#     plt.suptitle('\n{} Images'.format(case), y=0.95)
#     for i in range(size):
#         ax = fig.add_subplot(5, 2, i + 1, xticks=[], yticks=[])
#         paths[i] = paths[i].replace('\\', '/')
#
#         # gray scale image
#         img_string = tf.io.read_file(paths[i])
#         img = tf.image.decode_image(img_string, channels=1, dtype=tf.dtypes.float32, expand_animations=False)
#         img = tf.image.resize(img, [64, 256])
#         img = tf.image.grayscale_to_rgb(img)
#
#         # original images
#         # img = tf.image.decode_image(img_string, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
#         # img = tf.image.resize(img, [64, 256])
#
#         original = labels[i]
#         predicted = predictions[i]
#         ax.imshow(img)
#         ax.set_title(f"Original: {original}, Predicted: {predicted}")
#
#     plt.axis('off')
#     plt.show()

def main():

    testloader = TestLoader(configs)
    print('[Info] Load data from {}'.format(configs.val_list))

    checkpoint = torch.load(configs.model_path, map_location=torch.device('cpu'))

    model = Model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(configs.model_path))

    print('# Model Params = {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    tester = Tester(model, testloader)
    if configs.vert_test:
        tester.vert_val()
    else:
        tester.val()


if __name__== '__main__':
    # main()
    performance_analysis()
