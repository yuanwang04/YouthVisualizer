from celebA import *
from visualizer import *

"""
UW CSE 455 Final Project Youth Visualizer

The main program for visualizing some layers, activations, and deep dreams
for the CelebA data using the darkNet. 

Yuan Wang & Jiajie Shi. 2021-6-7
"""

IMAGE_PATH = './data/shijiajie1.jpg'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = './checkpoint/celebA_cp15.cp'
CELEB_A_IDX = 19
INPUT_TYPE = 'test'
# INPUT_TYPE = 'color'
# INPUT_TYPE = 'image'
DEEP_DREAM_YOUNG = 0

print('DEVICE: ' + DEVICE)


def main(celebA_idx=CELEB_A_IDX, input_type=INPUT_TYPE):
    ########## data ##########
    # load celebA test data
    print('loading data')
    celeb_data = get_celebA_data(test_shuffle=True, test_idxs=[celebA_idx], test_batch_size=1)
    print('data loaded')

    ########## input ##########
    # load input
    print('loading input')
    inputs = None
    if not input_type or input_type == 'test':
        itr = iter(celeb_data['test'])
        inputs, label = itr.__next__()
    elif input_type == 'image':
        inputs = get_inputs_from_file(IMAGE_PATH)
    elif input_type == 'color':
        inputs = get_color_input(channel=(0, 1, 2), size=(400, 400))

    # show the input
    # converted = tensor2img(img)
    # plt.imshow(converted)
    # plt.show()

    inputs = inputs.to(DEVICE)
    print('input loaded')

    ########## network ##########
    print('load net')
    net = CelebNet()
    checkpoint = torch.load(CHECKPOINT)
    net.load_state_dict(checkpoint['net'])
    net.to(DEVICE)
    print('net loaded')

    ########## layered activation ##########
    print('showing activated area at each layer')
    outputs = get_layered_result(inputs, net)
    imshow_grid(outputs[0][0])    # output at layer 1
    imshow_grid(outputs[1][0])    # output at layer 2
    imshow_grid(outputs[2][0])    # output at layer 3
    imshow_grid(outputs[3][0][:64])
    imshow_grid(outputs[4][0][:64])

    ########## rgb combined channels at each layer ##########
    print('showing rgb combined activation area at each layer')
    rgb_some_channel(inputs, net, layer_rgbs={1: (9, 13, 12), 2: (9, 3, 30), 3: (50, 53, 63)})

    ########## modify the input to activate some layer ##########
    print('showing images to activate some channel the most')
    layer = 2
    channel = 24
    schema = 0
    func = get_layer_funcs(net, channel={layer: channel})[layer]
    result = get_activation_inputs(inputs, func, rate=.002, epoch=200, collect_every=40, schema=schema)
    imshow_grid(result)
    modified_output = get_layered_result(result[5], net)
    imshow(outputs[layer-1][0][channel], modified_output[layer-1][0][channel])

    ########## deep dream ##########
    print('deep dream')
    outputs = deep_dream(inputs,
                         torch.tensor([DEEP_DREAM_YOUNG]).to(device=DEVICE),
                         net,
                         rate=.0001,
                         iter=1000,
                         collect_every=200)
    imshow_grid(outputs)


if __name__ == '__main__':
    main()
