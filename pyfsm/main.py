from miner import Miner, PlayState


if __name__ == '__main__':

	bob = Miner(PlayState.instance())

	while bob.is_life():
		bob.update()
