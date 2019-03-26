#define EPS 1e-9f

typedef struct bodyStruct { 
  float m, x, y, z,
        ax, ay, az, 
        vx, vy, vz; 
} bodyStruct;

void initialize_bodies(bodyStruct*, int);
void cal_acceleration(bodyStruct*, int);
void cal_velocity(bodyStruct*, int, float);