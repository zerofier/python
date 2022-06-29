
class State:

    def __init__(self, row=-1, col=-1):
        self.row = row
        self.col = col

    def __repr__(self):
        return f"<State: [{self.row}, {self.col}]>"

    def clone(self):
        return State(self.row, self.col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
