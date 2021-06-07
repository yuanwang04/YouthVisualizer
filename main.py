from celebA import *
from visualizer import *

"""
UW CSE 455 Final Project Youth Visualizer

The main program for visualizing some layers, activations, and deep dreams
for the CelebA data using the darkNet. 

Yuan Wang & Jiajie Shi. 2021-6-7
"""


def main(celebA_idx=231, input_type=''):
    ########## data ##########
    # load celebA test data
    print('loading data')
    celeb_data = get_celebA_data(test_shuffle=True, test_idxs=[celebA_idx], test_batch_size=1)
    print('data loaded')

    ########## input ##########
    # load input
    print('loading input')
    inputs = None
    if not input_type or input_type == 'celebA':
        itr = iter(celeb_data['test'])
        inputs, label = itr.__next__()
    elif input_type == 'image':
        inputs = get_inputs_from_file('./data/shijiajie.jpg')
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
    # outputs = get_layered_result(inputs, net)
    # imshow_grid(outputs[0][0])    # output at layer 1
    # imshow_grid(outputs[1][0])    # output at layer 2
    # imshow_grid(outputs[2][0])    # output at layer 3

    ########## rgb combined channels at each layer ##########
    print('showing rgb combined activation area at each layer')
    rgb_some_channel(inputs, net, layer_rgbs={1: (6, 5, 2), 2: (10, 3, 3), 3: (1, 3, 5)})

    ########## modify the input to activate some layer ##########
    print('showing images to activate some channel the most')
    func = get_layer_funcs(net, channel={2: 24})[2]
    result = get_activation_inputs(inputs, func, rate=.01, epoch=100, collect_every=20)
    imshow_grid(result)
    print('done')

    ########## deep dream ##########
    print('deep dream')
    outputs = deep_dream(inputs, torch.tensor([0]).to(device=DEVICE), net, rate=1, iter=100, collect_every=20)
    imshow_grid(outputs)


if __name__ == '__main__':
    main()
