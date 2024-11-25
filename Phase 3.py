from collections import defaultdict
import heapq
import numpy as np
from scipy.sparse import csr_matrix


# Collaborative Filtering Matrix
class CollaborativeFilteringMatrix:
    def __init__(self, num_users, num_products):
        self.num_users = num_users
        self.num_products = num_products
        self.matrix = csr_matrix((num_users, num_products), dtype=np.float32)

    def update_interaction(self, user_id, product_id, rating):
        """Update the matrix with a user's product rating."""
        self.matrix[user_id, product_id] = rating

    def get_user_ratings(self, user_id):
        """Retrieve all ratings given by a specific user."""
        return self.matrix[user_id].toarray()[0]

    def get_product_ratings(self, product_id):
        """Retrieve all ratings for a specific product."""
        return self.matrix[:, product_id].toarray().flatten()


# User Session Map
class UserSessionMap:
    def __init__(self):
        self.sessions = defaultdict(list)
        self.cache = {}

    def add_interaction(self, user_id, product_id):
        """Add a product interaction for a specific user."""
        self.sessions[user_id].append(product_id)
        if user_id in self.cache:
            self.cache.pop(user_id)  # Clear cache for updated user

    def get_user_session(self, user_id):
        """Retrieve all products interacted with by a specific user."""
        if user_id in self.cache:
            return self.cache[user_id]
        self.cache[user_id] = self.sessions[user_id]
        return self.cache[user_id]


# KDTree Implementation
class KDTreeNode:
    def __init__(self, point, left=None, right=None, product_id=None):
        self.point = point
        self.left = left
        self.right = right
        self.product_id = product_id


class KDTree:
    def __init__(self):
        self.root = None

    def insert(self, root, point, product_id=None, depth=0):
        """Insert a point into the KDTree."""
        if root is None:
            return KDTreeNode(point, product_id=product_id)
        k = len(point)
        axis = depth % k
        if point[axis] < root.point[axis]:
            root.left = self.insert(root.left, point, product_id, depth + 1)
        else:
            root.right = self.insert(root.right, point, product_id, depth + 1)
        return root

    def add_product(self, point, product_id):
        """Add a product's feature vector to the tree."""
        self.root = self.insert(self.root, point, product_id)

    def nearest_neighbor(self, root, target, depth=0, best=None):
        """Find the nearest neighbor for a given point."""
        if root is None:
            return best

        k = len(target)
        axis = depth % k

        # Update best
        if best is None or self.distance(root.point, target) < self.distance(best.point, target):
            best = root

        # Recursively search the tree
        next_branch = None
        opposite_branch = None
        if target[axis] < root.point[axis]:
            next_branch = root.left
            opposite_branch = root.right
        else:
            next_branch = root.right
            opposite_branch = root.left

        best = self.nearest_neighbor(next_branch, target, depth + 1, best)

        # Check if we need to explore the opposite branch
        if abs(target[axis] - root.point[axis]) < self.distance(best.point, target):
            best = self.nearest_neighbor(opposite_branch, target, depth + 1, best)

        return best

    @staticmethod
    def distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


# Example usage
if __name__ == "__main__":
    # Collaborative Filtering Example
    num_users, num_products = 5, 5
    cf_matrix = CollaborativeFilteringMatrix(num_users, num_products)
    cf_matrix.update_interaction(0, 0, 5)  # User 0 rates Product 0 as 5
    cf_matrix.update_interaction(1, 2, 3)  # User 1 rates Product 2 as 3
    print("User 0 Ratings:", cf_matrix.get_user_ratings(0))
    print("Product 2 Ratings:", cf_matrix.get_product_ratings(2))

    # User Session Example
    user_sessions = UserSessionMap()
    user_sessions.add_interaction(0, 101)
    user_sessions.add_interaction(0, 102)
    user_sessions.add_interaction(1, 103)
    print("User 0 Session:", user_sessions.get_user_session(0))
    print("User 1 Session:", user_sessions.get_user_session(1))

    # KDTree Example
    kdtree = KDTree()
    kdtree.add_product([1.0, 2.0], product_id=201)
    kdtree.add_product([3.0, 4.0], product_id=202)
    kdtree.add_product([5.0, 6.0], product_id=203)
    target_point = [2.0, 3.0]
    nearest = kdtree.nearest_neighbor(kdtree.root, target_point)
    print("Nearest Product to", target_point, "is Product ID:", nearest.product_id)
