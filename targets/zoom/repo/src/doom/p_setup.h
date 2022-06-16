//
// Copyright(C) 1993-1996 Id Software, Inc.
// Copyright(C) 2005-2014 Simon Howard
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// DESCRIPTION:
//   Setup a game, startup stuff.
//


#ifndef __P_SETUP__
#define __P_SETUP__

#include "doomdef.h"
#include "d_player.h"
#include "m_fixed.h"
#include "tables.h"

#define TANGOFUZZ_FEEDBACK_SHM  "/tangofuzz_feedback"
#define TANGOFUZZ_FEEDBACK_SIZE sizeof(tf_feedback_t)

typedef struct {
  float x;
  float y;
  float z;
} __attribute__((packed)) tf_location_t;

typedef struct {
  tf_location_t player_location;
  float         player_angle;

  playerstate_t player_state;
  int           health;                   // copied from player->mo->health
  int           armor_points;

  boolean       cards[NUMCARDS];
  int           weapon_owned[NUMWEAPONS];
  int           ammo[NUMAMMO];

  boolean       attacker_valid;
  tf_location_t attacker_location;

  boolean       did_secret;

  boolean       can_activate;
  float         tic_rate;
  boolean       floor_is_lava;
  boolean       secret_sector;

  boolean       pickup_valid;
  tf_location_t pickup_location;
} __attribute__((packed)) tf_feedback_t;

extern tf_feedback_t *tf_feedback;

// NOT called by W_Ticker. Fixme.
void
P_SetupLevel
( int		episode,
  int		map,
  int		playermask,
  skill_t	skill);

// Called by startup code.
void P_Init (void);

#endif
