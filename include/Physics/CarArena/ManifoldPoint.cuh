#pragma once

#include "../../State/Workspace.cuh"

struct ManifoldPoint
{
    Vec3 carPos;
    Vec3 worldPos;
    Vec3 worldNormal;
    float dist;
    float friction;
    float restitution;
    float impulse;
    float impulseTan;
    int lifetime;
};

CARL_D CARL_FI ManifoldPoint readManifoldPoint(
    const CarTriManifold& manifold,
    int idx)
{
    return {
        manifold.carPos[idx],
        manifold.worldPos[idx],
        manifold.worldNormal[idx],
        __ldg(&manifold.dist[idx]),
        __ldg(&manifold.friction[idx]),
        __ldg(&manifold.restitution[idx]),
        __ldg(&manifold.impulse[idx]),
        __ldg(&manifold.impulseTan[idx]),
        __ldg(&manifold.lifetime[idx])
    };
}

CARL_D CARL_FI void writeManifoldPoint(
    CarTriManifold& manifold,
    const ManifoldPoint& point,
    int idx)
{
    manifold.carPos[idx] = point.carPos;
    manifold.worldPos[idx] = point.worldPos;
    manifold.worldNormal[idx] = point.worldNormal;
    manifold.dist[idx] = point.dist;
    manifold.friction[idx] = point.friction;
    manifold.restitution[idx] = point.restitution;
    manifold.impulse[idx] = point.impulse;
    manifold.impulseTan[idx] = point.impulseTan;
    manifold.lifetime[idx] = point.lifetime;
}
