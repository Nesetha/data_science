class ReIdentifier:

    def __init__(self):
        self.visitors = []
        self.next_visitor_id = 1

    def identify(self, color, build, height):

        for visitor in self.visitors:

            if (
                visitor["color"] == color and
                visitor["build"] == build and
                abs(visitor["height"] - height) < 60 
            ):
                return visitor["visitor_id"], False

        visitor_id = self.next_visitor_id

        self.visitors.append({
            "visitor_id": visitor_id,
            "color": color,
            "build": build,
            "height": height
        })

        self.next_visitor_id += 1

        return visitor_id,True