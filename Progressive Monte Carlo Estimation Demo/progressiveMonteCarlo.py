import math
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
from gpytoolbox import ray_mesh_intersect, read_mesh, per_face_normals, \
    per_vertex_normals  # for ray-mesh intersection queries

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
        for ray origins and directions; gpytoolbox expects the
        number of ray origins to be equal to the number of ray
        directions (our code handles the cases where rays either
        all share the same origin or all share the same direction.)
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
        Computes an array of 3D locations given parametric
        distances along the rays
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


# abstraction for every scene object
class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return


# sphere objects for our scene
class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params):
        """
        Initializes a sphere object with its radius, position and diffuse albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        super().__init__()

    def intersect(self, rays):

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

# triangle mesh objects for our scene
class Mesh(Geometry):
    def __init__(self, filename, brdf_params):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        ### BEGIN SOLUTION
        # [TODO] replace the next line with your code for Phong normal interpolation
        self.face_normals = per_face_normals(self.v, self.f, unit_norm = True)
        self.per_vertex_normals = per_vertex_normals(self.v, self.f)
        ### END SOLUTION
        super().__init__()

    def intersect(self, rays):

        print('geometry')
        print('self.v')
        print(self.v)

        print('self.f')
        print(self.f)

        hit_normals = np.array([np.inf, np.inf, np.inf])
        hit_distances, triangle_hit_ids, barys = ray_mesh_intersect(rays.Os, rays.Ds, self.v,
                                                                    self.f, use_embree=True)
        ### BEGIN SOLUTION
        # [TODO] replace the next line with your code for Phong normal interpolation
        print('Bari')
        print(barys.shape)

        temp_normals_face = self.face_normals[triangle_hit_ids]

        temp_normals = []

        # Iterating through barycentric coordinates and generating
        # Phong interpolated normals for each pixel
        p = 0

        toBeMult = self.per_vertex_normals[self.f[triangle_hit_ids]]
        barysExtended = np.reshape(np.repeat(barys, 3), toBeMult.shape)

        # normalsDotDirArr = np.sum(normals * ray_directions_arr, axis = 1)

        temp_normals = np.sum(barysExtended * toBeMult, axis = 1)

        # while p < len(barys):
        #     triangle_id = triangle_hit_ids[p]
        #     if triangle_id != -1:
        #         temp_normals.append(normalize(barys[p][0] * self.per_vertex_normals[self.f[triangle_id][0]] + barys[p][1] * self.per_vertex_normals[self.f[triangle_id][1]] + barys[p][2] * self.per_vertex_normals[self.f[triangle_id][2]]))
        #     else:
        #         temp_normals.append([np.inf, np.inf, np.inf])
        #     p = p + 1

        ### END SOLUTION

        temp_normals = np.array(temp_normals)

        return hit_distances, temp_normals


# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, COSINE_SAMPLING = range(2)


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
        """ Adds a list of geometries to the scene. """
        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter=False):
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
        h_stretch_factor = math.tan(np.deg2rad(self.fov/2))
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


        ### END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        print('Ray Scene Intersection.')
        ### BEGIN SOLUTION

        # Initial current hits array. This way, if there is no hit, distance is np.inf
        currentHits = np.full((scene.w * scene.h, ), np.inf)
        currentNorms = np.full((scene.w * scene.h, 3), 0)
        # currentNorms = currentNorms.astype(object)
        # print(currentNorms)

        # Calculate hits for each sphere. In each iteration, only keep new values if they are lower than old values.
        # This solves primary visibiliy problem
        for spr in self.geometries:
            oldHits = currentHits
            newHits, newHitNormals = spr.intersect(rays)
            newHits[newHits < 0] = np.inf
            currentHits = np.minimum(newHits, oldHits)

             # Basically look at the changes between old distance and the new distances. If it has changed (i.e value is false), use this iteration's normals for that position
            useNewNormalsNot = currentHits == oldHits

            # Extend useNewNormalsNot to match the shape of currentNorms
            useNewNormalsNotExtended = np.reshape(np.repeat(useNewNormalsNot, 3), currentNorms.shape)

            currentNorms = np.where(useNewNormalsNotExtended == False, newHitNormals, currentNorms)

        # Initial id array. This way, if there is no hit, id is 0
        ids = np.zeros((self.w * self.h, 1))

        hit_ids = np.array(ids).astype(int)
        hit_ids = hit_ids - 1
        hit_distances = currentHits
        hit_normals = np.array(currentNorms)

        ### BEGIN SOLUTION
        ### END SOLUTION

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING, num_samples=1):

        ############## UNCOMMENT CODE BELOW FOR TEST 1 & 2

        # shadow_ray_o_offset = 1e-6
        #
        # # vectorized primary visibility test
        # distances, normals, ids = self.intersect(eye_rays)
        #
        # normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
        #                    normals, np.array([0, 0, 0]))
        #
        # hit_points = eye_rays(distances)
        #
        # # CAREFUL! when ids == -1 (i.e., no hit), you still get valid BRDF parameters!
        # brdf_params = np.array([obj.brdf_params for obj in self.geometries])[ids]
        #
        # # initialize the output "image" (i.e., vector; still needs to be reshaped)
        # L = np.zeros(normals.shape, dtype=np.float64)
        #
        # L = np.abs(normals)
        # L = L.reshape((self.h, self.w, 3))
        # return L

        ############## UNCOMMENT CODE ABOVE TO TEST DEL 1 & 2

        shadow_ray_o_offset = 1e-6

        # vectorized primary visibility test
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        distancesForIntersect = np.reshape(np.repeat(distances, 3), eye_rays.Os.shape)
        intersectionPoints = eye_rays.Os + eye_rays.Ds * distancesForIntersect

        # Offset intersection points along the normal
        intersectionPoints = intersectionPoints + shadow_ray_o_offset * normals

        # CAREFUL! when ids == -1 (i.e., no hit), you still get valid BRDF parameters!
        brdf_params = np.array([obj.brdf_params for obj in self.geometries])[ids]

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        ### BEGIN SOLUTION

        ## Generating array of canonical variables
        epOneArr = np.random.rand(self.w * self.h, )
        epTwoArr = np.random.rand(self.w * self.h, )

        # PDF based on full spherical sampling
        pdf = 1 / (4 * np.pi)

        # Calculating parameters for random ray shooting based on canonical random vars
        phi = 2 * np.pi * epTwoArr
        wz = 2 * epOneArr - 1
        r = np.sqrt(1 - np.square(wz))
        wx = r * np.cos(phi)
        wy = r * np.sin(phi)

        # Making a matrix of ray directions
        ray_directions_arr = np.array([wx, wy, wz]).T

        # Shooting out rays from the surface of the sphere/rabbit
        # To calculate visibility functions
        raysBundle = Rays(intersectionPoints, ray_directions_arr)
        hits_shadow, normals_shadow, ids_shadow = self.intersect(raysBundle)

        # Make np.inf (no intersections) into 1 in the hits array.
        # If there is a non inf value, then ray hit something, thus set it to 0 (not visible)
        hits = np.where(hits_shadow == np.inf, 1, 0)

        visibilityArr = hits

        placeHolderAlbedo = brdf_params[0]

        # Calculating the result piece by piece
        normalsDotDirArr = np.sum(normals * ray_directions_arr, axis = 1)
        maxNormDotDirAndZero = np.maximum(0, normalsDotDirArr)
        res = visibilityArr * maxNormDotDirAndZero
        res = res / pdf

        # Tiling albedo to match shape of final result
        albedo = np.tile(placeHolderAlbedo, (self.w * self.h, 1))

        # Tiling resArr to match shape of final result
        resArr = np.reshape(np.repeat(res, 3), albedo.shape)

        result = resArr * albedo
        result = result / np.pi

        overall = np.full(result.shape, 0)

        # Multiplying Lr with each light in the scene and adding up results
        for light in scene.lights:
            overall = overall + (light["color"] * result)
            overall = np.where(normals == 0, light["color"], overall)

        overall = overall

        overall = overall.reshape((self.h, self.w, 3))
        return overall

    def progressive_render_display(self, jitter=False, total_spp=20, spppp=1,
                                   sampling_type=UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)
        overall = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        # number of rendering iterations needed to obtain
        # (at least) our total desired spp
        progressive_iters = int(np.ceil(total_spp / spppp))

        i = 1.0
        while i - 1 < progressive_iters:
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            plt.title(f"current spp: {i} of {progressive_iters}")
            L = self.render(vectorized_eye_rays, sampling_type, spppp)
            overall = (overall + L)
            overallAverage = overall / i
            plt.imshow(overallAverage)
            # image_data.set_data(overall)
            i = i + 1
            plt.pause(0.001)  # add a tiny delay between rendering passes
            plt.savefig(f"render-{progressive_iters * spppp}spp.png")

            plt.show(block=False)

if __name__ == "__main__":
    enabled_tests = [False, False, True, False]

    #########################################################################
    ### 1 TESTS Eye Ray Anti Aliasing and Progressive Rendering
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9 / np.pi, 0.9 / np.pi, 0.9 / np.pi]))
        bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
                              brdf_params=np.array([0.9 / np.pi, 0.9 / np.pi, 0.9 / np.pi]))
        scene.add_geometries([bunny_sphere])
        scene.add_geometries([sphere])

        # no-AA
        # scene.progressive_render_display(jitter=False)

        # with AA
        scene.progressive_render_display(jitter=True)

    #########################################################################
    ### 2 TESTS Mesh Intersection and Phong Normal Interpolation
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        bunny = Mesh("bunny-446.obj",
                     brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])
        scene.add_geometries([bunny])

        # render a 1spp, no AA jittering image
        scene.progressive_render_display(jitter=False, total_spp=1, spppp=1)

    ###########################################################################
    ### 3 TESTS Ambient Occlusion with Uniform Importance Sampling
    ###########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])

        # DEBUG: start with a simpler scene before replacing your spherical bunny with an actual bunny
        # bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
        #                       brdf_params=np.array([0.9, 0.9, 0.9]))
        # scene.add_geometries([bunny_sphere])
        bunny = Mesh("bunny-446.obj",
                     brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([bunny])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=1,
                                         sampling_type=UNIFORM_SAMPLING)

    #########################################################################################
    ### 4 TESTS Ambient Occlusion with Cosine Importance Sampling
    #########################################################################################
    if enabled_tests[3]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])

        # DEBUG: start with a simpler scene before replacing your spherical bunny with an actual bunny
        bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
                              brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([bunny_sphere])
        # bunny = Mesh("bunny-446.obj",
        #              brdf_params=np.array([0.9, 0.9, 0.9]))
        # scene.add_geometries([bunny])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=1,
                                         sampling_type=COSINE_SAMPLING)
