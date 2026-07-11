class Arkiv:
    def __init__(self):
        self.arkiv = None

    def add(self, solution):
        if self.arkiv is None:
            self.arkiv = solution
        self.arkiv.append(solution)

    def get(self):
        return self.arkiv