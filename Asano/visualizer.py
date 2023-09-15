'''
学習結果の表示
各エポック時点でのテスト結果，平均損失と正解率を表示する
'''

import matplotlib.pyplot as plt

def visualizer(e, c, l):
    fig, ax1 = plt.subplots()
    ax1.plot(e, c, label='Accuracy', linestyle='-', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0,1)
    plt.grid(True)  # グリッドを表示


    # 2つ目のy軸に対するプロット
    ax2 = ax1.twinx()
    ax2.plot(e, l, label='Avg loss', linestyle='--', color='red')
    ax2.set_ylabel('Avg Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(bottom=0)

    # 凡例の表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper left')
    return fig