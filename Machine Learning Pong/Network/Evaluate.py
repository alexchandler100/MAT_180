from Pong.Game import Game


def evaluate(nn, draw, max_steps):
    finished = False
    offset = 10e+10
    outputs = []
    game = Game((1000, 600), draw=draw)
    running = 0
    time_steps = 0
    while running == 0:
        time_steps += 1
        inputs = game.getCords()
        output = nn.calc(inputs=[inputs[0], inputs[2], inputs[3]])[0]
        # print(output, nn.calc(inputs=[inputs[0], inputs[2], inputs[3]]))
        if output == 0:
            game.left_paddle.paddleMoveUp(-1)
        else:
            game.left_paddle.paddleMoveDown(1)
        game.gameStep()
        running = game.right_player
        outputs.append(output)
        if inputs[1] == -360:
            offset = abs(inputs[0] - inputs[2])
        if time_steps == max_steps:
            finished = True
            break
    if draw:
        for turtle in game.sc.turtles():
            turtle.reset()
            turtle.clear()
        game.sc.clear()
        game.sc.reset()
        game.sc.bye()
        del game
    return time_steps - offset, finished

