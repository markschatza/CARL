#pragma once

#include "../Physics/CarArena/ManifoldPoint.cuh"

// Decision-boundary snapshots for equally sized teams, in recorded world coordinates.
// Only persistent solver state is needed: broad/narrow candidates and the
// car/car manifold are regenerated before use each tick.
struct ReplayWheel {
    int source, contact;
    float latFriction, lonFriction;
};
struct ReplayCar {
    Vec3 pos, vel, ang, cen, imp;
    Quat rot;
    int isDemoed, contactCar, hitTick, contactTick, arenaTick;
    float demoTime, bumpCooldown;
    CarControls controls;
    CarInternalState internal;
    float brakeFactor, brakeFactorPrev, throttleVal, rawThrottle, handbrakeVal;
    float steerAngle, steerAnglePrev, engineDrive, engineDrivePrev;
    Vec3 jumpImpulse, angularImpulse;
    ReplayWheel wheels[NUM_WHEELS];
    int manifoldCount;
    ManifoldPoint manifold[MAX_CAR_MANIFOLD_POINTS];
};
template<int Cars>
struct TeamReplayLane {
    int tick, seed, sourceLane, episodeTicks, lastTouch, arenaTick;
    GoalState goal;
    Vec3 pos, vel, ang, imp, integrationPos, integrationVel, exportedPos, exportedVel;
    int integrationValid;
    float pads[NUM_BOOST_PADS];
    ReplayCar cars[Cars];
};
using ReplayLane = TeamReplayLane<2>;
template<int Cars>
constexpr int replayWords = sizeof(TeamReplayLane<Cars>) / sizeof(int32_t);
constexpr int REPLAY_WORDS = replayWords<2>;
static_assert(sizeof(TeamReplayLane<2>) % sizeof(int32_t) == 0);
static_assert(sizeof(TeamReplayLane<4>) % sizeof(int32_t) == 0);

template<bool restore, class T>
CARL_D CARL_FI void replayField(T& snapshot, T& live)
{
    if constexpr (restore) live = snapshot;
    else snapshot = live;
}

template<bool restore, int Cars>
__global__ void transferReplay(GameState* s, Workspace* ws, const int64_t* indices,
                               TeamReplayLane<Cars>* rows, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || indices[i] < 0 || indices[i] >= s->nSim) return;
    const int lane = static_cast<int>(indices[i]);
    auto& r = rows[i];
    if constexpr (restore) {
        s->replayOrigin[lane] = {r.tick - s->tickCount, r.seed, r.sourceLane + 1};
    } else {
        const auto origin = s->replayOrigin[lane];
        r.tick = s->laneTick(lane);
        r.seed = origin.lanePlusOne ? origin.seed : s->seed;
        r.sourceLane = origin.lanePlusOne ? origin.lanePlusOne - 1 : lane;
    }
    replayField<restore>(r.episodeTicks, s->episodeTicks[lane]);
    replayField<restore>(r.lastTouch, s->lastBallTouchTicks[lane]);
    replayField<restore>(r.arenaTick, s->ballArenaContactTick[lane]);
    replayField<restore>(r.goal, s->goals[lane]);
    auto& b = s->ball;
    replayField<restore>(r.pos, b.pos[lane]);
    replayField<restore>(r.vel, b.vel[lane]);
    replayField<restore>(r.ang, b.ang[lane]);
    replayField<restore>(r.imp, b.imp[lane]);
    replayField<restore>(r.integrationValid, b.integrationValid[lane]);
    if (r.integrationValid) {
        replayField<restore>(r.integrationPos, b.integrationPos[lane]);
        replayField<restore>(r.integrationVel, b.integrationVel[lane]);
        replayField<restore>(r.exportedPos, b.exportedPos[lane]);
        replayField<restore>(r.exportedVel, b.exportedVel[lane]);
    }
    for (int p = 0; p < NUM_BOOST_PADS; ++p)
        replayField<restore>(r.pads[p], s->boostPadCooldowns[lane * NUM_BOOST_PADS + p]);
    auto& cars = s->cars;
    auto& susp = ws->susp;
    for (int seat = 0; seat < Cars; ++seat) {
        const int c = lane * Cars + seat;
        auto& car = r.cars[seat];
        replayField<restore>(car.pos, cars.pos[c]);
        replayField<restore>(car.vel, cars.vel[c]);
        replayField<restore>(car.ang, cars.ang[c]);
        replayField<restore>(car.cen, cars.cen[c]);
        replayField<restore>(car.imp, cars.imp[c]);
        replayField<restore>(car.rot, cars.rot[c]);
        replayField<restore>(car.controls, cars.controls[c]);
        replayField<restore>(car.internal, cars.internal[c]);
        replayField<restore>(car.isDemoed, cars.isDemoed[c]);
        replayField<restore>(car.demoTime, cars.demoRespawnTimer[c]);
        if constexpr (restore) cars.carContactIdx[c] =
            car.contactCar < 0 ? -1 : lane * Cars + (car.contactCar % Cars);
        else car.contactCar = cars.carContactIdx[c] < 0 ? -1 : cars.carContactIdx[c] % Cars;
        replayField<restore>(car.bumpCooldown, cars.carContactCooldown[c]);
        replayField<restore>(car.hitTick, cars.ballHitTick[c]);
        replayField<restore>(car.contactTick, cars.ballContactTick[c]);
        replayField<restore>(car.arenaTick, cars.arenaContactTick[c]);
        replayField<restore>(car.brakeFactor, susp.brakeFactor[c]);
        replayField<restore>(car.brakeFactorPrev, susp.brakeFactorPrev[c]);
        replayField<restore>(car.throttleVal, susp.throttleVal[c]);
        replayField<restore>(car.rawThrottle, susp.rawThrottle[c]);
        replayField<restore>(car.handbrakeVal, susp.handbrakeVal[c]);
        replayField<restore>(car.steerAngle, susp.steerAngle[c]);
        replayField<restore>(car.steerAnglePrev, susp.steerAnglePrev[c]);
        replayField<restore>(car.engineDrive, susp.engineDrive[c]);
        replayField<restore>(car.engineDrivePrev, susp.engineDrivePrev[c]);
        replayField<restore>(car.jumpImpulse, susp.jumpImpulse[c]);
        replayField<restore>(car.angularImpulse, susp.angularImpulse[c]);
        for (int w = 0; w < NUM_WHEELS; ++w) {
            const int j = c * NUM_WHEELS + w;
            auto& wheel = car.wheels[w];
            replayField<restore>(wheel.source, cars.wheelSource[j]);
            replayField<restore>(wheel.contact, cars.wheelContact[j]);
            replayField<restore>(wheel.latFriction, susp.latFrictionPrev[j]);
            replayField<restore>(wheel.lonFriction, susp.lonFrictionPrev[j]);
        }
        auto& manifold = ws->ctMan;
        if constexpr (restore) {
            manifold.count[c] = min(max(car.manifoldCount, 0), MAX_CAR_MANIFOLD_POINTS);
            cars.internal[c].respawnIdx = min(max(cars.internal[c].respawnIdx, 0), 3);
        } else car.manifoldCount = manifold.count[c];
        for (int j = 0; j < manifold.count[c]; ++j) {
            const int idx = c * MAX_CAR_MANIFOLD_POINTS + j;
            if constexpr (restore) writeManifoldPoint(manifold, car.manifold[j], idx);
            else car.manifold[j] = readManifoldPoint(manifold, idx);
        }
    }
}
