import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
from gpytoolbox import read_mesh, per_vertex_normals, per_face_normals  # just used to load a mesh and compute per-vertex normals

def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; they must have the same
        size for gpytoolbox
        """
        if Os.shape[0] != Ds.shape[0]:
            if Ds.shape[0] == 1:
                self.Os = np.copy(Os)
                self.Ds = np.copy(Os)
                self.Ds[:, :] = Ds[:, :]
            if Os.shape[0] == 1:
                self.Ds = np.copy(Ds)
                self.Os = np.copy(Ds)
                self.Os[:, :] = Os[:, :]
        else:
            self.Os = np.copy(Os)
            self.Ds = np.copy(Ds)

    def __call__(self, t):
        """
        Computes an array of 3D locations given the distances
        to the points.
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return


def get_bary_coords(intersection, tri):
    denom = area(tri[:, 0], tri[:, 1], tri[:, 2])
    alpha_numerator = area(intersection, tri[:, 1], tri[:, 2])
    beta_numerator = area(intersection, tri[:, 0], tri[:, 2])
    alpha = alpha_numerator / denom
    beta = beta_numerator / denom
    gamma = 1 - alpha - beta
    barys = np.vstack((alpha, beta, gamma)).transpose()
    barys = np.where(np.isnan(barys), 0, barys)
    return barys


def area(t0, t1, t2):
    n = np.cross(t1 - t0, t2 - t0, axis=1)
    return np.linalg.norm(n, axis=1) / 2


def ray_mesh_intersect(origin, dir, tri):
    intersection = np.ones_like(dir) * -1
    intersection[:, 2] = np.Inf
    dir = dir[:, None]

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]  # (num_triangles, 3)
    s = origin[:, None] - tri[:, 0][None]
    s1 = np.cross(dir, e2)
    s2 = np.cross(s, e1)
    s1_dot_e1 = np.sum(s1 * e1, axis=2)
    results = np.ones((dir.shape[0], tri.shape[0])) * np.Inf

    if (s1_dot_e1 != 0).sum() > 0:
        coefficient = np.reciprocal(s1_dot_e1)
        alpha = coefficient * np.sum(s1 * s, axis=2)
        beta = coefficient * np.sum(s2 * dir, axis=2)
        cond_bool = np.logical_and(
            np.logical_and(
                np.logical_and(0 <= alpha, alpha < 1),
                np.logical_and(0 <= beta, beta < 1)
            ),
            np.logical_and(0 <= alpha + beta, alpha + beta < 1)
        )  # (num_rays, num_tri)
        e1_expanded = np.tile(e1[None], (dir.shape[0], 1, 1))  # (num_rays, num_tri, 3)
        dot_temp = np.sum(s1[cond_bool] * e1_expanded[cond_bool], axis=1)  # (num_rays,)
        results_cond1 = results[cond_bool]
        cond_bool2 = dot_temp != 0

        if cond_bool2.sum() > 0:
            coefficient2 = np.reciprocal(dot_temp)
            e2_expanded = np.tile(e2[None], (dir.shape[0], 1, 1))  # (num_rays, num_tri, 3)
            t = coefficient2 * np.sum(s2[cond_bool][cond_bool2] *
                                      e2_expanded[cond_bool][cond_bool2],
                                      axis=1)
            results_cond1[cond_bool2] = t
        results[cond_bool] = results_cond1
    results[results <= 0] = np.Inf
    hit_id = np.argmin(results, axis=1)
    min_val = np.min(results, axis=1)
    hit_id[min_val == np.Inf] = -1
    return min_val, hit_id

class Mesh(Geometry):
    def __init__(self, filename, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        self.Le = Le
        self.face_normals = per_face_normals(self.v, self.f, unit_norm=True)
        self.per_vertex_normals = per_vertex_normals(self.v, self.f)
        super().__init__()

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])

        hit_distances, triangle_hit_ids = ray_mesh_intersect(rays.Os, rays.Ds, self.v[self.f])
        intersections = rays.Os + hit_distances[:, None] * rays.Ds
        tris = self.v[self.f[triangle_hit_ids]]
        barys = get_bary_coords(intersections, tris)

        # Vectorized phong interpolation
        toBeMult = self.per_vertex_normals[self.f[triangle_hit_ids]]
        barysExtended = np.reshape(np.repeat(barys, 3), toBeMult.shape)
        temp_normals = np.sum(barysExtended * toBeMult, axis=1)

        return hit_distances, temp_normals


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()

    def intersect(self, rays):
        print('sphere_intersect')
        """
               Intersect the sphere with a bundle of rays, and compute the
               distance between the hit point on the sphere surface and the
               ray origins. If a ray did not intersect the sphere, set the
               distance to np.inf.
        """

        # If there is only one ray direction, tile it to match the dimenstions of ray origins array. Else leave it unchanged
        if(len(rays.Ds) == 1):
          rayDirections = np.tile(rays.Ds, (len(rays.Os), 1))
        else:
          rayDirections = rays.Ds

        rayOrigins = rays.Os

        sphereToRay = rayOrigins - self.c

        a = np.sum(rayDirections * rayDirections, axis = 1)
        b = 2 * np.sum(rayDirections * sphereToRay, axis = 1)
        c = np.sum(sphereToRay * sphereToRay, axis = 1) - self.r * self.r
        discrim = b * b - 4 * c
        distances = np.where(discrim < 0, np.inf, np.minimum((-b + np.sqrt(discrim)) / 2, (-b - np.sqrt(discrim)) / 2))

        distancesForIntersect = np.reshape(np.repeat(distances, 3), rays.Os.shape)
        intersections = rayOrigins + rayDirections * distancesForIntersect
        normals = intersections - self.c

        normals_squared_sum = np.sum(normals * normals, axis=1)
        normals_sum_underroot = np.sqrt(normals_squared_sum)
        normals_sum_underroot_reshaped = np.reshape(np.repeat(normals_sum_underroot, 3), normals.shape)
        normals_normalized = normals / normals_sum_underroot_reshaped

        return distances, normals_normalized


# Enumerate the different importance sampling strategies we will implement
IMPLICIT_UNIFORM_SAMPLING, EXPLICIT_UNIFORM_SAMPLING, IMPLICIT_BRDF_SAMPLING, EXPLICIT_LIGHT_BRDF_SAMPLING = range(4)


class Scene(object):
    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_geometries(self, geometries):
        """
        Adds a list of geometries to the scene.

        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([geometries[i]])

        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter=False):
        print('generate_eye_rays')
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """

        # Generating initial array according to height and weight
        x_coords_ndc = np.array(np.arange(0, self.w, 1))
        y_coords_ndc = np.array(np.arange(0, self.h, 1))

        # Adding offset of 0.5 so that rays pass through middle of the pixel
        x_coords_ndc = (x_coords_ndc + 0.5) / self.w
        y_coords_ndc = (y_coords_ndc + 0.5) / self.h

        # If jittered
        if jitter:
            x_jitter = np.random.uniform(low=0, high=0.5, size=(self.w,))
            y_jitter = np.random.uniform(low=-0, high=0.5, size=(self.h,))

            x_jitter = x_jitter / self.w
            y_jitter = y_jitter / self.h

            x_coords_ndc = x_coords_ndc + x_jitter
            y_coords_ndc = y_coords_ndc + y_jitter


        # Offsetting coordinates so that middle is 0,0
        x_coords_ndc = (2 * x_coords_ndc) - 1
        y_coords_ndc = 1 - (2 * y_coords_ndc)

        # Calculating stretch factor according to aspect ratio and fov
        h_stretch_factor = np.tan(np.deg2rad(self.fov/2))
        w_stretch_factor = (self.w / self.h) * h_stretch_factor

        # Applying stretch factor to bring NDC x anx y into camera space
        x_coords_cam = x_coords_ndc * w_stretch_factor
        y_coords_cam = y_coords_ndc * h_stretch_factor

        # Repeat x coordinates array for each y coordinate to construct final x array
        x_coords_cam = np.tile(x_coords_cam, len(y_coords_ndc))

        # Repeat each element in y, len(x) times to construct y array
        y_coords_cam = np.repeat(y_coords_cam, len(x_coords_ndc))

        # Repeat 1, len(x) times to construct x array
        z_coords_cam = np.repeat([1], len(x_coords_cam))

        """ Generating M"""

        # Calculate Xc, Yc and Zc vectors
        ZCVector = self.at - self.eye
        XCVector = np.cross(self.up, ZCVector)
        YCVector = np.cross(ZCVector, XCVector)

        ## Normalise
        ZCVector = ZCVector / np.linalg.norm(ZCVector)
        XCVector = XCVector / np.linalg.norm(XCVector)
        YCVector = YCVector / np.linalg.norm(YCVector)

        # Appending 0s to make 4D
        ZCVector = np.append(ZCVector, 0)
        XCVector = np.append(XCVector, 0)
        YCVector = np.append(YCVector, 0)

        # Extract eye coorindate and append 1 to make 4 d
        eyeForMVector = np.copy(self.eye)
        eyeForMVector = np.append(eyeForMVector, 1)

        # Forming M matrix
        M = np.array([XCVector, YCVector, ZCVector, eyeForMVector])
        M = M.T

        # Stack x, y and z arrays
        ToBeTransformedPointsMatrix = np.array([x_coords_cam, y_coords_cam, z_coords_cam, z_coords_cam])

        # Multiply M with stacked points to transform coordinates into world space
        transformedPointsMatrix = np.matmul(M, ToBeTransformedPointsMatrix)

        # Cut out last row of 1s to get 3D coordinates
        transformedPointsMatrix = transformedPointsMatrix[:3]

        # transpose to group together x, y, z point coordinates
        transformedPointsMatrix = transformedPointsMatrix.T

        # subtract origin from final point to get eye vector directions
        directions = transformedPointsMatrix - self.eye

        # normalise eye vector directions
        directions = directions / np.linalg.norm(directions, axis = 1, keepdims = True)

        # Extend origins array to match direction array in size
        origins = np.tile(self.eye, (len(directions), 1))

        return Rays(origins, directions)
        ### END CODE

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """

        # Initial current hits array. This way, if there is no hit, distance is np.inf
        currentHits = np.full((scene.w * scene.h,), np.inf)
        currentNorms = np.full((scene.w * scene.h, 3), 0)
        currentIds = np.full((scene.w * scene.h,), 0)

        # Calculate hits for each sphere. In each iteration, only keep new values if they are lower than old values.
        # This solves primary visibility problem

        i = 0
        for spr in self.geometries:
            oldHits = currentHits
            newHits, newHitNormals = spr.intersect(rays)
            newHits[newHits < 0] = np.inf
            currentHits = np.minimum(newHits, oldHits)

            # Basically look at the changes between old distance and the new distances. If it has changed (i.e value is false), use this iteration's normals for that position
            useNewNormalsNot = currentHits == oldHits
            currentIds = np.where(useNewNormalsNot == False, i, currentIds)

            # Extend useNewNormalsNot to match the shape of currentNorms
            useNewNormalsNotExtended = np.reshape(np.repeat(useNewNormalsNot, 3), currentNorms.shape)

            currentNorms = np.where(useNewNormalsNotExtended == False, newHitNormals, currentNorms)
            i = i + 1

        # Initial id array. This way, if there is no hit, id is 0
        ids = np.zeros((self.w * self.h, 1))
        hit_ids = currentIds
        hit_distances = currentHits
        hit_normals = np.array(currentNorms)

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, num_bounces=3, sampling_type=IMPLICIT_BRDF_SAMPLING):

        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),np.array([0,0,0,1])[np.newaxis,:]))[ids]
        np.concatenate((np.array([obj.brdf_params for obj in self.geometries]), np.array([0, 0, 0, 1])[np.newaxis, :]))[
            ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array([0, 0, 0])[np.newaxis, :]))[ids]
        L_e_objects = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array([0, 0, 0])[np.newaxis, :]))
        objects = np.concatenate((np.array([obj for obj in self.geometries]), np.array([-1])))
        lightsArray = np.array([obj for obj in self.lights])
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]), np.array([-1])))[ids]


        if sampling_type==IMPLICIT_BRDF_SAMPLING:

            shadow_ray_o_offset = 1e-6

            # vectorized primary visibility test
            distances, normals, ids = self.intersect(eye_rays)

            normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                               normals, np.array([0, 0, 0]))

            # Contruct active matrix, first start with initial array where everything is on
            activeMatrix = np.full((self.w * self.h, ), 1)

            # If there is no intersection, make pixel active indicator = 0, else leave it as is
            activeMatrix = np.where(distances == np.inf, 0, activeMatrix)

            # Construct matrix composed of Le of intersections
            LeMatrix = L_e_objects[ids]

            # Split each row up
            light_transposed = LeMatrix.T
            l1 = light_transposed[0:1]
            l2 = light_transposed[1:2]
            l3 = light_transposed[2:3]

            l1 = l1.T
            l2 = l2.T
            l3 = l3.T

            l1 = np.reshape(l1, (l1.shape[0],))
            l2 = np.reshape(l2, (l2.shape[0],))
            l3 = np.reshape(l3, (l3.shape[0],))

            # If each row is 0, then this is not a light. If even one of them is non zero, this is a light hit
            isLightHit = np.where(np.logical_and(l1 == 0, l2 == 0, l3 == 0), 0, 1)

            # If there is light intersection, make pixel active indicator = 0, else leave it as is
            activeMatrix = np.where(isLightHit == 1, 0, activeMatrix)

            # Calculating intersection points
            distancesForIntersect = np.reshape(np.repeat(distances, 3), eye_rays.Os.shape)
            intersectionPoints = eye_rays.Os + eye_rays.Ds * distancesForIntersect

            # Offset intersection points along the normal
            intersectionPoints = intersectionPoints + shadow_ray_o_offset * normals

            overall = np.full((self.w * self.h, 3), 0)

            for light in scene.lights:

                # Isolating the last element of brdf elements to see if it's glossy or diffuse
                alphaArr = brdf_params.T
                alphaArr = alphaArr[3:4]
                alphaArr = alphaArr.T

                # Generating array of canonical variables
                epOneArr = np.random.rand(self.w * self.h, )
                epTwoArr = np.random.rand(self.w * self.h, )

                # Calculating parameters for random ray shooting based on canonical random vars
                phi = 2 * np.pi * epTwoArr
                alphaArrWz = np.reshape(alphaArr, (len(alphaArr), ))
                wz = np.power(epOneArr, (1 / alphaArrWz))
                r = np.sqrt(1 - np.square(wz))
                wx = r * np.cos(phi)
                wy = r * np.sin(phi)

                # Making a matrix of ray directions
                ray_directions_arr = np.array([wx, wy, wz]).T

                ###### Transforming specular ray vectors into world space

                # Calculates wR
                wO = (eye_rays.Ds) * -1
                normalsDotWO = np.sum(normals * wO, axis=1)
                normalsDotWOExtended = np.reshape(np.repeat(normalsDotWO, 3), (normals.shape))
                normalsDotWOMultipliedByNormals = normalsDotWOExtended * normals
                wR = 2 * normalsDotWOMultipliedByNormals - wO

                # Calculating vectors for M matrix for transforming into world space
                transformMatrixZSpec = wR
                upMatrixSpec = np.random.rand(transformMatrixZSpec.shape[0], transformMatrixZSpec.shape[1])
                transformMatrixXSpec = np.cross(transformMatrixZSpec, upMatrixSpec)
                transformMatrixYSpec = np.cross(transformMatrixXSpec, transformMatrixZSpec)

                # Normalizing vectors
                transformMatrixXSpec = transformMatrixXSpec / np.linalg.norm(transformMatrixXSpec, axis=1, keepdims=True)
                transformMatrixZSpec = transformMatrixZSpec / np.linalg.norm(transformMatrixZSpec, axis=1, keepdims=True)
                transformMatrixYSpec = transformMatrixYSpec / np.linalg.norm(transformMatrixYSpec, axis=1, keepdims=True)

                # Creating transformation matrix M for specular reflections
                MSpec = np.stack((transformMatrixXSpec, transformMatrixYSpec, transformMatrixZSpec), axis=1)
                MSpec = MSpec

                # Multiplying rayDirs by Mspec to transform into world space for specular reflections
                ray_directions_arrSpec = np.einsum('ij,ijk->ik', ray_directions_arr, MSpec)
                ray_directions_arrSpec = ray_directions_arrSpec / np.linalg.norm(ray_directions_arrSpec, axis=1, keepdims=True)

                ###### Transforming diffuse ray vectors into world space

                # Creating transformation matrix M for diffuse reflections
                transformMatrixZDiffuse = normals
                upMatrixDiffuse = np.random.rand(transformMatrixZDiffuse.shape[0], transformMatrixZDiffuse.shape[1])
                transformMatrixXDiffuse = np.cross(transformMatrixZDiffuse, upMatrixDiffuse)
                transformMatrixYDiffuse = np.cross(transformMatrixXDiffuse, transformMatrixZDiffuse)

                # Normalizing vectors
                transformMatrixXDiffuse = transformMatrixXDiffuse / np.linalg.norm(transformMatrixXDiffuse, axis=1, keepdims=True)
                transformMatrixZDiffuse = transformMatrixZDiffuse / np.linalg.norm(transformMatrixZDiffuse, axis=1, keepdims=True)
                transformMatrixYDiffuse = transformMatrixYDiffuse / np.linalg.norm(transformMatrixYDiffuse, axis=1, keepdims=True)

                # Creating transformation matrix M for diffuse reflections
                MDiffuse = np.stack((transformMatrixXDiffuse, transformMatrixYDiffuse, transformMatrixZDiffuse), axis=1)

                # Multiplying rayDirs by Mspec to transform into world space for diffuse reflections
                ray_directions_arrDiffuse = np.einsum('ij,ijk->ik', ray_directions_arr, MDiffuse)
                ray_directions_arrDiffuse = ray_directions_arrDiffuse / np.linalg.norm(ray_directions_arrDiffuse, axis=1, keepdims=True)

                # Extend the alphaArr
                alphaArrMaskExtended = np.reshape(np.repeat(alphaArr, 3), ray_directions_arrDiffuse.shape)

                # decide which ray direction to use (spec/diffuse) based on alphaArr
                ray_directions_arr = np.where(alphaArrMaskExtended > 1, ray_directions_arrSpec, ray_directions_arrDiffuse)

                ###### Shooting out randomly generated rays

                # To calculate visibility functions
                raysBundle = Rays(intersectionPoints, ray_directions_arr)
                hits_shadow, normals_shadow, ids_shadow = self.intersect(raysBundle)

                # Make np.inf (no intersections) into 1 in the hits array.
                # If there is a non inf value, then ray hit something, thus set it to 0 (not visible)
                hits = np.where(hits_shadow == np.inf, 0, 1)

                # Check if the hit object is the current light. If it is, set it to visArr value. Else, 0
                # Consider skipping this entire part if this isn't true
                visibilityArr = np.where(objects[ids_shadow] == light, hits, 0)

                # Calculating the result piece by piece
                normalsDotDirArr = np.sum(normals * ray_directions_arr, axis=1)
                maxNormDotDirAndZero = np.maximum(0, normalsDotDirArr)

                # If last element of brdf element was 1, this is a diffuse surface
                diffuseMask = alphaArr == 1

                # Extending diffuseMask to fit albedoArr shape
                diffuseMask = np.reshape(diffuseMask.repeat(3), (self.w * self.h, 3))

                albedoArr = brdf_params.T
                albedoArr = albedoArr[0:3]
                albedoArr = albedoArr.T

                # Filtering out all non diffuse albedos. Set to 0 everything else so it contributes nothing
                diffuseAlbedoArr = np.where(diffuseMask, albedoArr, 0)
                diffuse_result = np.reshape(np.repeat(visibilityArr, 3), diffuseAlbedoArr.shape) * diffuseAlbedoArr

                # If last element of brdf element was 1, this is a spec surface
                specMask = alphaArr > 1

                specAlphaArr = np.where(specMask, alphaArr, 0)
                specAlphaArr = np.reshape(specAlphaArr.repeat(3), (self.w * self.h, 3))

                specMask = np.reshape(specMask.repeat(3), (self.w * self.h, 3))
                specAlbedoArr = np.where(specMask, albedoArr, 0)

                specResult = np.reshape(np.repeat(visibilityArr, 3), specAlphaArr.shape) * np.reshape(np.repeat(maxNormDotDirAndZero, 3), specAlphaArr.shape) * specAlbedoArr

                specAndDiffResult = specResult + diffuse_result

                overall = overall + (light.Le * specAndDiffResult)

                # # Here, we filter by activeMatrix
                filterRes = overall

                recursiveRayDirections = ray_directions_arr
                recursiveIntersectionPoints = intersectionPoints

            print('OVERALL BRIGHTNESS')
            print(np.sum(overall))

            for i in range(num_bounces):

                recursiveRayBundle = Rays(recursiveIntersectionPoints, recursiveRayDirections)

                distances, normals, ids = self.intersect(recursiveRayBundle)
                brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                              np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]

                # If there is no intersection, make pixel active indicator = 0, else leave it as is
                activeMatrix = np.where(distances == np.inf, 0, activeMatrix)

                # Construct matrix composed of Le of intersections
                LeMatrix = L_e_objects[ids]

                # Split each row up
                light_transposed = LeMatrix.T
                l1 = light_transposed[0:1]
                l2 = light_transposed[1:2]
                l3 = light_transposed[2:3]

                l1 = l1.T
                l2 = l2.T
                l3 = l3.T

                l1 = np.reshape(l1, (l1.shape[0],))
                l2 = np.reshape(l2, (l2.shape[0],))
                l3 = np.reshape(l3, (l3.shape[0],))

                # If each row is 0, then this is not a light. If even one of them is non zero, this is a light hit
                isLightHit = np.where(np.logical_and(l1 == 0, l2 == 0, l3 == 0), 0, 1)

                # If there is light intersection, make pixel active indicator = 0, else leave it as is
                activeMatrix = np.where(isLightHit == 1, 0, activeMatrix)

                # Calculating new recursive intersection points
                recursiveDistancesForIntersect = np.reshape(np.repeat(distances, 3), recursiveRayBundle.Os.shape)
                recursiveIntersectionPoints = recursiveRayBundle.Os + recursiveRayBundle.Ds * recursiveDistancesForIntersect

                # Offset intersection points along the normal
                recursiveIntersectionPoints = recursiveIntersectionPoints + shadow_ray_o_offset * normals

                firstPass = 0
                for light in scene.lights:
                    # Isolating the last element of brdf elements to see if it's glossy or diffuse
                    alphaArr = brdf_params.T
                    alphaArr = alphaArr[3:4]
                    alphaArr = alphaArr.T

                    # Generating array of canonical variables
                    epOneArr = np.random.rand(self.w * self.h, )
                    epTwoArr = np.random.rand(self.w * self.h, )

                    # Calculating parameters for random ray shooting based on canonical random vars
                    phi = 2 * np.pi * epTwoArr
                    alphaArrWz = np.reshape(alphaArr, (len(alphaArr),))
                    wz = np.power(epOneArr, (1 / alphaArrWz))
                    r = np.sqrt(1 - np.square(wz))
                    wx = r * np.cos(phi)
                    wy = r * np.sin(phi)

                    # Making a matrix of ray directions
                    recursiveRayDirections = np.array([wx, wy, wz]).T

                    # Calculates wR
                    wO = (recursiveRayBundle.Ds) * -1
                    normalsDotWO = np.sum(normals * wO, axis=1)
                    normalsDotWOExtended = np.reshape(np.repeat(normalsDotWO, 3), (normals.shape))
                    normalsDotWOMultipliedByNormals = normalsDotWOExtended * normals
                    wR = 2 * normalsDotWOMultipliedByNormals - wO

                    # Calculating vectors for M matrix for transforming into world space
                    transformMatrixZSpec = wR
                    upMatrixSpec = np.random.rand(transformMatrixZSpec.shape[0], transformMatrixZSpec.shape[1])
                    transformMatrixXSpec = np.cross(transformMatrixZSpec, upMatrixSpec)
                    transformMatrixYSpec = np.cross(transformMatrixXSpec, transformMatrixZSpec)

                    # Normalizing vectors
                    transformMatrixXSpec = transformMatrixXSpec / np.linalg.norm(transformMatrixXSpec, axis=1,
                                                                                 keepdims=True)
                    transformMatrixZSpec = transformMatrixZSpec / np.linalg.norm(transformMatrixZSpec, axis=1,
                                                                                 keepdims=True)
                    transformMatrixYSpec = transformMatrixYSpec / np.linalg.norm(transformMatrixYSpec, axis=1,
                                                                                 keepdims=True)

                    # Creating transformation matrix M for specular reflections
                    MSpec = np.stack((transformMatrixXSpec, transformMatrixYSpec, transformMatrixZSpec), axis=1)
                    MSpec = MSpec

                    # Multiplying rayDirs by Mspec to transform into world space for specular reflections
                    recursiveRayDirectionsSpec = np.einsum('ij,ijk->ik', recursiveRayDirections, MSpec)
                    recursiveRayDirectionsSpec = recursiveRayDirectionsSpec / np.linalg.norm(recursiveRayDirectionsSpec,
                                                                                             axis=1,
                                                                                             keepdims=True)

                    # Creating transformation matrix M for diffuse reflections
                    transformMatrixZDiffuse = normals
                    upMatrixDiffuse = np.random.rand(transformMatrixZDiffuse.shape[0],
                                                     transformMatrixZDiffuse.shape[1])
                    transformMatrixXDiffuse = np.cross(transformMatrixZDiffuse, upMatrixDiffuse)
                    transformMatrixYDiffuse = np.cross(transformMatrixXDiffuse, transformMatrixZDiffuse)

                    # Normalizing vectors
                    transformMatrixXDiffuse = transformMatrixXDiffuse / np.linalg.norm(transformMatrixXDiffuse,
                                                                                       axis=1, keepdims=True)
                    transformMatrixZDiffuse = transformMatrixZDiffuse / np.linalg.norm(transformMatrixZDiffuse,
                                                                                       axis=1, keepdims=True)
                    transformMatrixYDiffuse = transformMatrixYDiffuse / np.linalg.norm(transformMatrixYDiffuse,
                                                                                       axis=1, keepdims=True)

                    # Creating transformation matrix M for diffuse reflections
                    MDiffuse = np.stack((transformMatrixXDiffuse, transformMatrixYDiffuse, transformMatrixZDiffuse),
                                        axis=1)

                    # Multiplying rayDirs by Mspec to transform into world space for diffuse reflections
                    recursiveRayDirectionsDiffuse = np.einsum('ij,ijk->ik', recursiveRayDirections, MDiffuse)
                    recursiveRayDirectionsDiffuse = recursiveRayDirectionsDiffuse / np.linalg.norm(
                        recursiveRayDirectionsDiffuse, axis=1, keepdims=True)

                    # Extend the alphaArr
                    alphaArrMaskExtended = np.reshape(np.repeat(alphaArr, 3), recursiveRayDirectionsDiffuse.shape)

                    # decide which ray direction to use (spec/diffuse) based on alphaArr
                    recursiveRayDirections = np.where(alphaArrMaskExtended > 1, recursiveRayDirectionsSpec,
                                                      recursiveRayDirectionsDiffuse)

                    # To calculate visibility functions
                    raysBundle = Rays(recursiveIntersectionPoints, recursiveRayDirections)
                    hits_shadow, normals_shadow, ids_shadow = self.intersect(raysBundle)

                    # Make np.inf (no intersections) into 1 in the hits array.
                    # If there is a non inf value, then ray hit something, thus set it to 0 (not visible)
                    hits = np.where(hits_shadow == np.inf, 0, 1)

                    # Check if the hit object is the current light. If it is, set it to visArr value. Else, 0
                    # Consider skipping this entire part if this isn't true
                    visibilityArr = np.where(objects[ids_shadow] == light, hits, 0)

                    # Calculating the result piece by piece
                    normalsDotDirArr = np.sum(normals * recursiveRayDirections, axis=1)
                    maxNormDotDirAndZero = np.maximum(0, normalsDotDirArr)

                    # If last element of brdf element was 1, this is a diffuse surface
                    diffuseMask = alphaArr == 1

                    # Extending diffuseMask to fit albedoArr shape
                    diffuseMask = np.reshape(diffuseMask.repeat(3), (self.w * self.h, 3))

                    albedoArr = brdf_params.T
                    albedoArr = albedoArr[0:3]
                    albedoArr = albedoArr.T

                    # Filtering out all non diffuse albedos. Set to 0 everything else so it contributes nothing
                    diffuseAlbedoArr = np.where(diffuseMask, albedoArr, 0)

                    diffuse_result = np.reshape(np.repeat(visibilityArr, 3),
                                                diffuseAlbedoArr.shape) * diffuseAlbedoArr

                    print('diffuse_result INSIDE')
                    print(np.sum(diffuse_result))

                    # If last element of brdf element was 1, this is a spec surface
                    specMask = alphaArr > 1

                    specAlphaArr = np.where(specMask, alphaArr, 0)
                    specAlphaArr = np.reshape(specAlphaArr.repeat(3), (self.w * self.h, 3))

                    specMask = np.reshape(specMask.repeat(3), (self.w * self.h, 3))
                    specAlbedoArr = np.where(specMask, albedoArr, 0)

                    print('specAlbedoArr INSIDE')
                    print(np.sum(specAlbedoArr))

                    # if(firstPass == 0):
                    #     specAlbedoArr = specAlbedoArr * specAlbedoArrNew

                    maxNormDotDirAndZero = np.nan_to_num(maxNormDotDirAndZero, nan=0)

                    print('np.repeat(maxNormDotDirAndZero, 3) INSIDE')
                    print(np.sum(np.repeat(maxNormDotDirAndZero, 3)))
                    print(maxNormDotDirAndZero)


                    print('np.reshape(np.repeat(visibilityArr, 3), specAlphaArr.shape)')
                    print(np.sum(np.reshape(np.repeat(visibilityArr, 3), specAlphaArr.shape)))

                    maxNormDotDirAndZero = np.where(maxNormDotDirAndZero == np.inf, 0, maxNormDotDirAndZero)


                    specResult = np.reshape(np.repeat(visibilityArr, 3), specAlphaArr.shape) * np.reshape(
                        np.repeat(maxNormDotDirAndZero, 3), specAlphaArr.shape) * specAlbedoArr

                    print('specResult INSIDE')
                    print(np.sum(specResult))

                    specAndDiffResult = specResult + diffuse_result

                    print('specAndDiffResult INSIDE')
                    print(np.sum(specAndDiffResult))

                    overall = filterRes + (light.Le * specAndDiffResult)

                    filterRes = overall

                    print('OVERALL BRIGHTNESS INSIDE')
                    print(np.sum(overall))

                    # Here, we filter by activeMatrix
                    activeMatrixExtended = np.reshape(np.repeat(activeMatrix, 3), overall.shape)

                    filterRes = np.where(activeMatrixExtended == 1, overall, filterRes)
                    # firstPass = firstPass + 1

            # overall = np.where(normals == 0, light.Le, overall)
            overall = np.where(np.logical_and(L_e != np.array([0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, overall)
            overall = overall.reshape((self.h, self.w, 3))
            return overall

    def progressive_render_display(self, jitter=False, total_spp=20, num_bounces=3,
                                   sampling_type=IMPLICIT_BRDF_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)
        overall = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(overall)

        progressive_iters = int(total_spp)

        i = 1.0
        while i - 1 < progressive_iters:
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            plt.title(f"current spp: {i} of {progressive_iters}")
            L = self.render(vectorized_eye_rays, sampling_type)
            overall = (overall + L)
            overallAverage = overall / i
            overallAverage = np.clip(overallAverage, 0, 1)
            # plt.imshow(overallAverage)
            image_data.set_data(overallAverage)
            i = i + 1
            plt.pause(0.0000001)  # add a tiny delay between rendering passes
            plt.savefig(f"render-{progressive_iters * 1}--{sampling_type}spp--numbounce--{num_bounces}.png")

            plt.show(block=False)


if __name__ == "__main__":
    enabled_tests = [True, True, True]
    enable_deliverables = [True, False] 

    #########################################################################
    ### Test Case 1: Default Cornell Box Scene
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        scene = Scene(w=int(128 / 2), h=int(128 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        ### Test 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            # scene.progressive_render_display(total_spp=500, jitter=True, num_bounces=2,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=501, jitter=True, num_bounces=3,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            # scene.progressive_render_display(total_spp=502, jitter=True, num_bounces=4,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Test 2: Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    ### Test Case 3: Scene with decreasing light size (constant power)
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=int(128 / 2), h=int(128 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        ### Test 4: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            # scene.progressive_render_display(total_spp=503, jitter=True, num_bounces=2,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=504, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=505, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Test 5: Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    ### Test Case 6: Scene with different BRDFs
    #########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        scene = Scene(w=int(128 / 2), h=int(128 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 50])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1000]))
        ])

        #########################################################################
        ### Test 7: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            # scene.progressive_render_display(total_spp=506, jitter=True, num_bounces=2,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=507, jitter=True, num_bounces=3,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            # scene.progressive_render_display(total_spp=508, jitter=True, num_bounces=4,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Test 8: Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)