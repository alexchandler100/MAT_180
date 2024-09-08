class Game:
    def __init__(self, size, AI="match", draw=True):
        self.draw = draw
        self.size = size
        self.AI = AI
        self.vel = 1

        # Create Paddles and ball
        self.left_paddle = Paddle((-size[0] / 2 + 100, 0), draw)
        self.right_paddle = Paddle((size[0] / 2 - 100, 0), draw)
        self.pong = Pong(draw=draw)

        # Create Scores
        self.left_player = 0
        self.right_player = 0

        if draw:
            import turtle
            turtle.TurtleScreen._RUNNING = True
            sc = turtle.Screen()
            sc.title("Pong")
            sc.bgcolor("black")
            sc.setup(width=size[0], height=size[1])
            score = turtle.Turtle()
            score.speed(0)
            score.color("blue")
            score.penup()
            score.hideturtle()
            score.goto(0, size[1] / 2 - 40)
            score.write("Left_player : 0    Right_player: 0", align="center", font=("Courier", 24, "normal"))
            self.score = score
            sc.onkeypress(self.left_paddle.paddleMoveUp, "e")
            sc.onkeypress(self.left_paddle.paddleMoveDown, "x")
            sc.onkeypress(self.right_paddle.paddleMoveUp, "Up")
            sc.onkeypress(self.right_paddle.paddleMoveDown, "Down")
            sc.listen()
            self.sc = sc

    def gameStep(self):
        draw = self.draw
        if draw:
            self.sc.update()
        self.pong.move()
        x, y = self.pong.getCords()
        w, h = self.size

        if self.AI == "match":
            self.right_paddle.paddleMoveUp(self.pong.velocity[1])
        elif self.AI == "constant":
            self.right_paddle.paddleMoveUp(self.vel)
            if self.right_paddle.getY() < -h / 2 or self.right_paddle.getY() > h / 2:
                self.vel *= -1

        # Ball hits top or bottom

        if y > h / 2 - 20:
            self.pong.setY(h / 2 - 20)
            self.pong.scaleVelocity([1, -1])
        if y < - h / 2 + 20:
            self.pong.setY(- h / 2 + 20)
            self.pong.scaleVelocity([1, -1])

        # Ball is scored

        if x > w / 2:
            self.pong.setX(0), self.pong.setY(0)
            self.pong.scaleVelocity([-1, -1])
            self.left_player += 1
            if draw:
                self.score.clear()
                self.score.write("Left_player : {}    Right_player: {}".format(self.left_player, self.right_player),
                                 align="center", font=("Courier", 24, "normal"))
        if x < -w / 2:
            self.pong.setX(0), self.pong.setY(0)
            self.pong.scaleVelocity([-1, -1])
            self.right_player += 1
            if draw:
                self.score.clear()
                self.score.write("Left_player : {}    Right_player: {}".format(self.left_player, self.right_player),
                                 align="center", font=("Courier", 24, "normal"))

        # Ball hits Paddle
        ly, ry = self.left_paddle.getY(), self.right_paddle.getY()
        # if -w / 2 + 100 < x < -w / 2 + 140:
        #     print(ly, y)
        if w / 2 - 100 > x > w / 2 - 140 and ry + 40 > y > ry - 40:
            self.pong.setX(w / 2 - 140)
            self.pong.scaleVelocity([-1, 1])
        if -w / 2 + 100 < x < -w / 2 + 140 and ly + 40 > y > ly - 40:
            self.pong.setX(-w / 2 + 140)
            self.pong.scaleVelocity([-1, 1])

    def play(self):
        while True:
            self.gameStep()

    def getCords(self):
        ball = self.pong.getCords()
        return self.left_paddle.getY(), ball[0], ball[1], self.left_paddle.getY() - ball[1]


class Paddle:
    def __init__(self, place, draw):
        self.y = place[1]
        self.x = place[0]
        self.draw = draw
        if draw:
            import turtle
            turtle.TurtleScreen._RUNNING = True
            pad = turtle.Turtle()
            pad.shape("square")
            pad.color("white")
            pad.shapesize(stretch_wid=6, stretch_len=2)
            pad.penup()
            pad.goto(place[0], place[1])
            self.pad = pad

    # This could be done with 1 function, but in order for the game to work in manual mode it needs to be called like this
    # Moves the paddle by v, positive v move the paddle up, negative move it down.
    def paddleMoveUp(self, v=5):
        self.y += v
        if self.draw:
            self.pad.sety(self.y)

    def paddleMoveDown(self, v=-5):
        self.y += v
        if self.draw:
            self.pad.sety(self.y)

    # Returns the y coordinate of the paddle
    def getY(self):
        return self.y


class Pong:
    def __init__(self, draw, speed=None):
        if speed is None:
            speed = [-1, -1]
        self.x = 0
        self.y = 0
        self.velocity = speed
        self.draw = draw
        if draw:
            import turtle
            turtle.TurtleScreen._RUNNING = True
            ball = turtle.Turtle()
            ball.shape("circle")
            ball.color("blue")
            ball.penup()
            ball.goto(self.x, self.y)
            ball.speed(40)
            self.ball = ball

    # Moves the pong by velocity every game step
    def move(self):
        v = self.velocity
        self.x += v[0]
        self.y += v[1]
        if self.draw:
            self.ball.setx(self.x), self.ball.sety(self.y)

    # Returns the x,y coordinate pair of the ball.
    def getCords(self):
        return self.x, self.y

    # Sets the y coordinate of the ball
    def setY(self, y):
        self.y = y
        if self.draw:
            self.ball.sety(y)

    # Sets the x coordinate of the ball
    def setX(self, x):
        self.x = x
        if self.draw:
            self.ball.setx(x)

    # Scales the velocity by the inputted vector
    def scaleVelocity(self, v):
        self.velocity[0] *= v[0]
        self.velocity[1] *= v[1]

# game = Game((1000, 500))
# game.play()
